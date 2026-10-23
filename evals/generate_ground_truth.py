import argparse
import logging
import os
import pathlib
import re

from azure.identity import AzureDeveloperCliCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from dotenv_azd import load_azd_env
from langchain_core.documents import Document as LCDocument
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, default_transforms
from rich.logging import RichHandler

from evaltools.ground_truth_files import (
    GROUND_TRUTH_BY_FILE_DIR,
    knowledge_graph_path,
    questions_path,
    remove_source_file,
    write_qa_pairs,
)

logger = logging.getLogger("ragapp")

root_dir = pathlib.Path(__file__).parent
by_file_dir = root_dir / GROUND_TRUTH_BY_FILE_DIR
combined_ground_truth_path = root_dir / "ground_truth.jsonl"


def get_azure_credential():
    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    if AZURE_TENANT_ID:
        logger.info("Setting up Azure credential using AzureDeveloperCliCredential with tenant_id %s", AZURE_TENANT_ID)
        azure_credential = AzureDeveloperCliCredential(tenant_id=AZURE_TENANT_ID, process_timeout=60)
    else:
        logger.info("Setting up Azure credential using AzureDeveloperCliCredential for home tenant")
        azure_credential = AzureDeveloperCliCredential(process_timeout=60)
    return azure_credential


def build_sourcefile_filter(source_files: list[str] | None) -> str | None:
    if not source_files:
        return None
    # OData string literals escape a single quote by doubling it
    return " or ".join("sourcefile eq '{}'".format(name.replace("'", "''")) for name in source_files)


def get_search_documents(azure_credential, num_search_documents=None, source_files=None) -> str:
    search_client = SearchClient(
        endpoint=f"https://{os.getenv('AZURE_SEARCH_SERVICE')}.search.windows.net",
        index_name=os.getenv("AZURE_SEARCH_INDEX"),
        credential=azure_credential,
    )
    all_documents = []
    if num_search_documents is None:
        logger.info("Fetching all document chunks from Azure AI Search")
        num_search_documents = 100000
    else:
        logger.info("Fetching %d document chunks from Azure AI Search", num_search_documents)
    search_filter = build_sourcefile_filter(source_files)
    if search_filter:
        logger.info("Only fetching chunks from: %s", ", ".join(source_files))
    response = search_client.search(search_text="*", filter=search_filter, top=num_search_documents).by_page()
    for page in response:
        page = list(page)
        all_documents.extend(page)
    if source_files and not all_documents:
        raise ValueError(
            f"No chunks found for {', '.join(source_files)}. "
            "Use the exact file name as indexed (for example 'employee_handbook.pdf'), without any folder path."
        )
    return all_documents


def generate_ground_truth_ragas(
    num_questions=200,
    num_search_documents=None,
    kg_file=None,
    source_files=None,
    output_path=combined_ground_truth_path,
    kg_output_path=root_dir / "ground_truth_kg.json",
    append=True,
):
    azure_credential = get_azure_credential()
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-06-01"
    azure_endpoint = f"https://{os.getenv('AZURE_OPENAI_SERVICE')}.openai.azure.com"
    azure_ad_token_provider = get_bearer_token_provider(
        azure_credential, "https://cognitiveservices.azure.com/.default"
    )
    generator_llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            openai_api_version=azure_openai_api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=azure_ad_token_provider,
            azure_deployment=os.getenv("AZURE_OPENAI_EVAL_DEPLOYMENT"),
            model=os.environ["AZURE_OPENAI_EVAL_MODEL"],
            validate_base_url=False,
        )
    )

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    generator_embeddings = LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            openai_api_version=azure_openai_api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=azure_ad_token_provider,
            azure_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT"),
            model=os.environ["AZURE_OPENAI_EMB_MODEL_NAME"],
        )
    )

    # Load or create the knowledge graph
    if kg_file:
        full_path_to_kg = root_dir / kg_file
        if not os.path.exists(full_path_to_kg):
            raise FileNotFoundError(f"Knowledge graph file {full_path_to_kg} not found.")
        logger.info("Loading existing knowledge graph from %s", full_path_to_kg)
        kg = KnowledgeGraph.load(full_path_to_kg)
    else:
        # Make a knowledge_graph from Azure AI Search documents
        search_docs = get_search_documents(azure_credential, num_search_documents, source_files)

        logger.info("Creating a RAGAS knowledge graph based off of %d search documents", len(search_docs))
        nodes = []
        for doc in search_docs:
            content = doc["content"]
            citation = doc["sourcepage"]
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": f"[[{citation}]]: {content}",
                    "document_metadata": {"citation": citation},
                },
            )
            nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

        logger.info("Using RAGAS to apply transforms to knowledge graph")
        transforms = default_transforms(
            documents=[LCDocument(page_content=doc["content"]) for doc in search_docs],
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )
        apply_transforms(kg, transforms)

        kg_output_path.parent.mkdir(parents=True, exist_ok=True)
        kg.save(kg_output_path)
        logger.info("Saved knowledge graph to %s", kg_output_path)

    logger.info("Using RAGAS knowledge graph to generate %d questions", num_questions)
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg)
    dataset = generator.generate(testset_size=num_questions, with_debugging_logs=True)

    qa_pairs = []
    for sample in dataset.samples:
        question = sample.eval_sample.user_input
        truth = sample.eval_sample.reference
        # Grab the citation in square brackets from the reference_contexts and add it to the truth
        citations = []
        for context in sample.eval_sample.reference_contexts:
            match = re.search(r"\[\[(.*?)\]\]", context)
            if match:
                citation = match.group(1)
                citations.append(f"[{citation}]")
        truth += " " + " ".join(citations)
        qa_pairs.append({"question": question, "truth": truth})

    logger.info("Writing %d QA pairs to %s", len(qa_pairs), output_path)
    write_qa_pairs(output_path, qa_pairs, append=append)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Generate ground truth data using AI Search index and RAGAS.")
    parser.add_argument("--numsearchdocs", type=int, help="Specify the number of search results to fetch")
    parser.add_argument(
        "--numquestions",
        type=int,
        help="Specify the number of questions to generate (per file when --sourcefile is given).",
        default=200,
    )
    parser.add_argument("--kgfile", type=str, help="Specify the path to an existing knowledge graph file")
    parser.add_argument(
        "--sourcefile",
        action="append",
        help=f"Generate questions from only this indexed file name, written to {GROUND_TRUTH_BY_FILE_DIR}/. "
        "Repeat to process several files, each into its own file.",
    )
    parser.add_argument(
        "--reusekg",
        action="store_true",
        help="With --sourcefile, reuse that file's saved knowledge graph instead of rebuilding it.",
    )
    parser.add_argument(
        "--remove",
        action="append",
        help=f"Delete this file's questions and knowledge graph from {GROUND_TRUTH_BY_FILE_DIR}/ "
        "and drop its questions from ground_truth.jsonl. Repeat for several files.",
    )

    args = parser.parse_args()

    if args.remove:
        for source_file in args.remove:
            deleted_paths, removed_count = remove_source_file(by_file_dir, combined_ground_truth_path, source_file)
            for path in deleted_paths:
                logger.info("Deleted %s", path)
            logger.info("Removed %d questions citing %s from %s", removed_count, source_file, combined_ground_truth_path)
        raise SystemExit(0)

    if args.sourcefile and args.kgfile:
        parser.error("--kgfile can't be combined with --sourcefile; use --reusekg to reuse a file's saved graph.")

    load_azd_env(override=os.getenv("LOADING_MODE_FOR_AZD_ENV_VARS") != "no-override")

    if args.sourcefile:
        for source_file in args.sourcefile:
            saved_kg_path = knowledge_graph_path(by_file_dir, source_file)
            reuse_kg = args.reusekg and saved_kg_path.exists()
            if args.reusekg and not reuse_kg:
                logger.info("No saved knowledge graph for %s yet, building one", source_file)
            generate_ground_truth_ragas(
                num_search_documents=args.numsearchdocs,
                num_questions=args.numquestions,
                kg_file=str(saved_kg_path) if reuse_kg else None,
                source_files=[source_file],
                output_path=questions_path(by_file_dir, source_file),
                kg_output_path=saved_kg_path,
                append=False,
            )
    else:
        generate_ground_truth_ragas(
            num_search_documents=args.numsearchdocs,
            num_questions=args.numquestions,
            kg_file=args.kgfile,
        )
