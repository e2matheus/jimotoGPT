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
        print(f"\n{fore.WHITE}Is there anything I can help you with?{style.RESET}\n")
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
        loadingMessage = "Loading..."
        print(f"{fore.BLACK}{loadingMessage}{style.RESET}")

        # Go up one line and move the cursor to the end of the line
        print(f"\033[1A\033[{len('Loading...')}C", end="")
        res = None
        try:
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        except Exception as e:
            print(f"\n{fore.RED}Error: {e}{style.RESET}\n")
            continue

        end = time.time()

        # Use end to detect if the answer was found.
        # The idea is to move the cursor to the beginning of the first line,
        # while removing all the characters from the answer, then to remove
        # the characters of "Loading..." as well
        if end - start > 0.5:
            # Count the number of characters in the answer
            numberOfCharacters = len(answer)

            # If the answer was not found, print a message and continue
            if numberOfCharacters == 0:
                print(f"\n{fore.MAGENTA}Sorry, I don't know the answer to that question.{style.RESET}\n")
                continue

            # Get the number of characters than can fit on the command line
            # depending on the size of the code editor window
            terminalSize = os.get_terminal_size().columns

            def removeLineCharacters(total):
                for i in range(total):
                    # Press backspace
                    print(f"\033[1D\033[K", end="")
                    # wait for 0.01 seconds
                    time.sleep(0.01)

            answerChunks = []

            # If the number of characters in the answer is greater than the number of
            # characters that can fit on the command line, store the answer in
            # multiple lines with an array of strings
            if numberOfCharacters > terminalSize:
                subanswer = [answer[i:i+terminalSize] for i in range(0, len(answer), terminalSize)]

                # Push chunk of the answer to answerChunks array
                for i in range(len(subanswer)):
                    answerChunks.append(subanswer[i])
            else:
                answerChunks.append(answer)

            # Go through all the answer chunks and use a function
            # to remove the number of characters in each chunk,
            # starting from the last chunk and goes to
            # the first chunk
            for i in range(len(answerChunks) - 1, -1, -1):
                removeLineCharacters(len(answerChunks[i]))

                # wait for 0.01 seconds
                time.sleep(0.01)

                # If the are more chucks to remove, go up one line
                if i > 0:
                    print(f"\033[1A", end="")

        # wait for 0.01 seconds before printing the answer
        time.sleep(0.01)

        # Print the answer after trimming the whitespace from it
        trimmedAnswer = answer.strip()
        print(f"\033[{len(loadingMessage)}D\033[K{style.RESET}{fore.WHITE}{trimmedAnswer}{style.RESET}")

        # wait for 0.02 seconds before the next question
        time.sleep(0.20)

        # Ask the user if they want to show the analysis to get the result
        canPrintResult = False

        print(f"\n{fore.BLACK}Do I show how I got to the answer?{style.RESET}\n")

        resultPrintquestions = [
            inquirer.Confirm('confirm', message=f"{fore.YELLOW}"),
        ]
        canPrintResult = inquirer.prompt(resultPrintquestions)['confirm']

        if canPrintResult:
            # Print the result
            questionHeader = "> Question:"
            print(f"{fore.CYAN}{questionHeader}{style.RESET}")
            print(f"{fore.GREY}{query}{style.RESET}")

            answerHeader = f"\n> Answer (took {round(end - start, 2)} s.):"
            print(f"{fore.CYAN}{answerHeader}{style.RESET}")
            print(f"{style.RESET}{fore.GREY}{trimmedAnswer}{style.RESET}")

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
