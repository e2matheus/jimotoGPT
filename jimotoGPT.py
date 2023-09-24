#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
import inquirer
from yachalk import chalk
import pycolors
from pycolors import fore, style

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Prepare the LLM
    print(f"{fore.GREY}Loading model...")
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    print(f"{style.RESET}{fore.WHITE}")

    # Interactive questions and answers
    while True:
        print(f"\n\n{fore.WHITE}Is there anything I can help you with?{style.RESET}\n")
        questions = [
            inquirer.Text('query', message=f"{fore.YELLOW}", validate=lambda _, x: len(x.strip()) > 0)
        ]
        query = inquirer.prompt(questions)['query']

        if query == "exit" or query == "quit":
            print(f"\n{fore.WHITE}Seems like that's it for now. Goodbye!{style.RESET}\n\n")
            break

        # Get the answer from the chain
        start = time.time()
        print(f"{style.RESET}{fore.WHITE}")
        res = qa(query)
        # formattedResult = f"{fore.GREEN}{res['result']}{style.RESET}"
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        questionHeader = "> Question:"
        print(f"\n\n{fore.CYAN}{questionHeader}{style.RESET}")
        print(f"{fore.GREY}{query}{style.RESET}")

        answerHeader = f"\n> Answer (took {round(end - start, 2)} s.):"
        print(f"{fore.CYAN}{answerHeader}{style.RESET}")
        print(f"{style.RESET}{fore.GREY}{answer}{style.RESET}")

        # Print the relevant sources used for the answer
        for document in docs:
            metadataSource = document.metadata["source"]
            print(f"\n{fore.BLUE}> {metadataSource}:{style.RESET}")
            print(f"{fore.GREY}{document.page_content}{style.RESET}")

        print(f"{style.RESET}{fore.WHITE}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='jimotoGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
