import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
import os
import pickle
import tempfile
import pandas as pd
import pdfplumber
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
import sys
import re
from dotenv import load_dotenv
from io import BytesIO
from io import StringIO
import datetime
import json
import openai
import re
from tqdm.auto import tqdm
from typing import List, Union
import zipfile

from langchain.agents import create_csv_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Langchain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# LLM wrapper
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
# Conversational memory
from langchain.memory import ConversationBufferWindowMemory


class Utilities:

    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or 
        from the user's input and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
            )
            if user_api_key:
                st.sidebar.success("API keys loaded", icon="🚀")

        return user_api_key
    
    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type=["pdf"], label_visibility="collapsed", accept_multiple_files = True)
        if uploaded_file is not None:

            def show_pdf_file(uploaded_file):
                file_container = st.expander("Your PDF file :")
                for i in range(len(uploaded_file)):
                    with pdfplumber.open(uploaded_file[i]) as pdf:
                        pdf_text = ""
                        for page in pdf.pages:
                            pdf_text += page.extract_text() + "\n\n"
                    file_container.write(pdf_text)
            
            file_extension = ".pdf" 

            if file_extension== ".pdf" : 
                show_pdf_file(uploaded_file)

        else:
            st.sidebar.info(
                "👆 Upload your PDF file to get started..!"
            )
            st.session_state["reset_chat"] = True

        #print(uploaded_file)
        return uploaded_file

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature,):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder()
        # Use RecursiveCharacterTextSplitter as the default and only text splitter
        splitter_type = "RecursiveCharacterTextSplitter"
        with st.spinner("Processing..."):
            #uploaded_file.seek(0)
            file = uploaded_file
            
            # Get the document embeddings for the uploaded file
            vectors = embeds.getDocEmbeds(file, "Docs")

            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature,vectors)
        st.session_state["ready"] = True

        return chatbot

    def count_tokens_agent(agent, query):
        """
        Count the tokens used by the CSV Agent
        """
        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f'Spent a total of {cb.total_tokens} tokens')

        return result

class Layout:
    
    def show_header(self):
        """
        Displays the header of the app
        """
        st.markdown(
            """
            <h1 style='text-align: center;'> Ask Anything: Your Personal AI Assistant</h1>
            """,
            unsafe_allow_html=True,
        )

    def show_api_key_missing(self):
        """
        Displays a message if the user has not entered an API key
        """
        st.markdown(
            """
            <div style='text-align: center;'>
                <h4>Enter your <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API key</a> to start conversation</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def prompt_form(self):
        """
        Displays the prompt form
        """
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area(
                "Query:",
                placeholder="Ask me anything about the document...",
                key="input",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(label="Send")
            
            is_ready = submit_button and user_input
        return is_ready, user_input


class Sidebar:

    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about():
        about = st.sidebar.expander("🧠 About")
        sections = [
            "#### Welcome to our AI Assistant, a cutting-edge solution to help you find the answers you need quickly and easily. Our AI Assistant is designed to provide you with the most relevant information from a variety of sources, including PDFs, CSVs, and web search.",
            "#### With our AI Assistant, you can ask questions on any topic, and our intelligent algorithms will search through our vast database to provide you with the most accurate and up-to-date information available. Whether you need help with a school assignment, are researching a topic for work, or simply want to learn something new, our AI Assistant is the perfect tool for you.",
        ]
        for section in sections:
            about.write(section)

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        model = st.selectbox(label="Model", options=self.MODEL_OPTIONS)
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
        )
        st.session_state["temperature"] = temperature
        
    def csv_agent_button(self, uploaded_file):
        st.session_state.setdefault("show_csv_agent", False)

    def show_options(self, uploaded_file):
        with st.sidebar.expander("🛠️ Tools", expanded=False):

            self.reset_chat_button()
            self.csv_agent_button(uploaded_file)
            # self.model_selector()
            # self.temperature_slider()
            st.session_state.setdefault("model", model_name)
            st.session_state.setdefault("temperature", temperature)

original_filename="Docs"
class Embedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeds(self, file, original_filename="Docs"):
        """
        Stores document embeddings using Langchain and FAISS
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        
        text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size = 2000,
                chunk_overlap  = 50,
                length_function = len,
            )
        file_extension = ".pdf" #get_file_extension(original_filename)


        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)  
            data = loader.load_and_split(text_splitter)
        
            
        embeddings = OpenAIEmbeddings()

        vectors = FAISS.from_documents(data, embeddings)
        os.remove(tmp_file_path)

        # Save the vectors to a pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            pickle.dump(vectors, f)


    def getDocEmbeds(self, file, original_filename):
        """
        Retrieves document embeddings
        """
        # Use RecursiveCharacterTextSplitter as the default and only text splitter
        splitter_type = "RecursiveCharacterTextSplitter"
        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(file)
        #st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=500,
                             overlap=0, split_method=splitter_type)
        embeddings = OpenAIEmbeddings()
        vectors = create_retriever(embeddings, splits, retriever_type="SIMILARITY SEARCH")
        return vectors

class ChatHistory:
    
    def __init__(self):
        self.history = st.session_state.get("history", [])
        st.session_state["history"] = self.history

    def default_greeting(self):
        return "Hey! 👋"

    def default_prompt(self, topic):
        return f"Hello ! Ask me anything about {topic} 🤗"

    def initialize_user_history(self):
        st.session_state["user"] = [self.default_greeting()]

    def initialize_assistant_history(self, uploaded_file):
        st.session_state["assistant"] = [self.default_prompt(original_filename)]

    def initialize(self, uploaded_file):
        if "assistant" not in st.session_state:
            self.initialize_assistant_history(original_filename)
        if "user" not in st.session_state:
            self.initialize_user_history()

    def reset(self, uploaded_file):
        st.session_state["history"] = []
        
        self.initialize_user_history()
        self.initialize_assistant_history(original_filename)
        st.session_state["reset_chat"] = False

    def append(self, mode, message):
        st.session_state[mode].append(message)

    def generate_messages(self, container):
        if st.session_state["assistant"]:
            with container:
                for i in range(len(st.session_state["assistant"])):
                    message(
                        st.session_state["user"][i],
                        is_user=True,
                        key=f"{i}_user",
                        avatar_style="big-smile",
                    )
                    message(st.session_state["assistant"][i], key=str(i), avatar_style="thumbs")

    def load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                self.history = f.read().splitlines()

    def save(self):
        with open(self.history_file, "w") as f:
            f.write("\n".join(self.history))


from langchain.chains.question_answering import load_qa_chain
class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors


    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow-up entry: {question}
        Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """You are a friendly conversational assistant, designed to answer questions and chat with the user from a contextual file.
        You receive data from a user's files and a question, you must help the user find the information they need. 
        Your answers must be user-friendly and respond to the user.
        You will get questions and contextual information.
        question: {question}
        =========
        context: {context}
        ======="""
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        retriever = self.vectors#.as_retriever()

        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=True)
        doc_chain = load_qa_chain(llm=llm, 
                                  
                                  prompt=self.QA_PROMPT,
                                  verbose=True,
                                  chain_type= "stuff"
                                  )

        chain = ConversationalRetrievalChain(
            retriever=retriever, combine_docs_chain=doc_chain, question_generator=question_generator, verbose=True, return_source_documents=True)


        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]

def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import UnstructuredPDFLoader
import PyPDF2
@st.cache_data
def load_docs(files):
    st.sidebar.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text




@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.sidebar.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=[" ", ",", "\n"])

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits
def doc_search(temperature):
    os.environ["SERPAPI_API_KEY"] = user_serpapi_key
    st.sidebar.success("Upload PDF To Chat With!", icon="👇")
    uploaded_file = st.sidebar.file_uploader("Upload PDF file here!", type="pdf", accept_multiple_files = True)
    if uploaded_file is None:
        st.warning("Upload PDF File first!!")
    else:
        search = SerpAPIWrapper()
        # Set up a prompt template which can interpolate the history
        template_with_history = """You are SearchGPT, a professional search engine who provides informative answers to users. Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin! Remember to give detailed, informative answers

        Previous conversation history:
        {history}

        New question: {input}
        {agent_scratchpad}"""
        def search_chroma(query):
                #result_docs = vectordb.similarity_search(query)
                retriever = db#.as_retriever(search_type="mmr") # db.similarity_search(query)
                retrieval_llm = ChatOpenAI(model_name=model_name, temperature=temperature, top_p=top_p, frequency_penalty=freq_penalty)
                # Initiate our LLM - default is 'gpt-3.5-turbo'
                llm = ChatOpenAI(model_name = model_name, temperature=temperature)
                podcast_retriever = RetrievalQA.from_chain_type(llm=retrieval_llm, chain_type="stuff", retriever=retriever)
                expanded_tools = [
                            Tool(
                                name = "Search",
                                func=search.run,
                                description="useful for when you need to answer questions about current events"
                            ),
                            Tool(
                                name = 'Knowledge Base',
                                func=podcast_retriever.run,
                                description="Useful for general questions about how to do things and for details on interesting topics. Input should be a fully formed question."
                            )
                        ]
                # Re-initialize the agent with our new list of tools
                prompt_with_history = CustomPromptTemplate(
                    template=template_with_history,
                    tools=expanded_tools,
                    input_variables=["input", "intermediate_steps", "history"]
                )
                llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
                multi_tool_names = [tool.name for tool in expanded_tools]
                multi_tool_agent = LLMSingleActionAgent(
                    llm_chain=llm_chain, 
                    output_parser=output_parser,
                    stop=["\nObservation:"], 
                    allowed_tools=multi_tool_names
                )
                multi_tool_memory = ConversationBufferWindowMemory(k=0)
                multi_tool_executor = AgentExecutor.from_agent_and_tools(agent=multi_tool_agent, tools=expanded_tools, verbose=True, memory=multi_tool_memory)
                output = multi_tool_executor.run(query)
                return output
        def get_text():
                input_text = st.text_input("", key="input")
                return input_text 
        def generate_response(prompt):
                completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens[model_name],
                top_p=top_p,
                frequency_penalty=freq_penalty,
                messages=[
                    {"role": "user", "content": prompt}
                ])
                response = completion.choices[0].message.content
                return response
        def prompt_form():
            """
            Displays the prompt form
            """
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Query:",
                    placeholder="Ask me anything about the document...",
                    key="input_",
                    label_visibility="collapsed",
                )
                submit_button = st.form_submit_button(label="Send")
                
                is_ready = submit_button and user_input
            return is_ready, user_input

        #layout.show_header()
        embeddings = OpenAIEmbeddings()
        # Use RecursiveCharacterTextSplitter as the default and only text splitter
        splitter_type = "RecursiveCharacterTextSplitter"
        loaded_text = load_docs(uploaded_file)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=500,
                             overlap=0, split_method=splitter_type)
        # Display the number of text chunks
        num_chunks = len(splits)
        st.sidebar.write(f"Number of text chunks: {num_chunks}")
        db = create_retriever(embeddings, splits, retriever_type="SIMILARITY SEARCH")
        st.write("Write your query here:💬")

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ['I am ready to help you sir']

        if 'past' not in st.session_state:
            st.session_state['past'] = ['Hey there!']
        #user_input = get_text()
        is_ready, user_input = prompt_form()
        #is_readyy = st.button("Send")
        if is_ready: # user_input:
            output = search_chroma(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state['generated']:

            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
def init():
    load_dotenv()
    st.set_page_config(layout="wide", page_icon="💬", page_title="AI Chatbot 🤖")

def main(temperature):
    # Initialize the app
    #init()

    # Instantiate the main components
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()

    layout.show_header()

    #user_api_key = utils.load_api_key()

    if not user_api_key:
        layout.show_api_key_missing()
    else:
        os.environ["OPENAI_API_KEY"] = user_api_key

        # search = st.sidebar.button("Web Search Chat")
        # if search:
        #     doc_search()

        uploaded_file = utils.handle_upload()

        if uploaded_file:
            # Initialize chat history
            history = ChatHistory()

            # Configure the sidebar
            sidebar.show_options(uploaded_file)

            try:
                chatbot = utils.setup_chatbot(
                    uploaded_file, st.session_state["model"], st.session_state["temperature"]
                )
                st.session_state["chatbot"] = chatbot

                if st.session_state["ready"]:
                    # Create containers for chat responses and user prompts
                    response_container, prompt_container = st.container(), st.container()

                    with prompt_container:
                        # Display the prompt form
                        is_ready, user_input = layout.prompt_form()

                        # Initialize the chat history
                        history.initialize(uploaded_file)

                        # Reset the chat history if button clicked
                        if st.session_state["reset_chat"]:
                            history.reset(uploaded_file)

                        if is_ready:
                            # Update the chat history and display the chat messages
                            history.append("user", user_input)
                            output = st.session_state["chatbot"].conversational_chat(user_input)
                            history.append("assistant", output)

                    history.generate_messages(response_container)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    sidebar.about()

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        # if not match:
        #     raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()

# Define a dictionary with the function names and their respective functions
functions = [
    "Chat with Docs",
    "Chat with Docs + Web Search"
]

st.set_page_config(layout="wide", page_icon="💬", page_title="AI Chatbot 🤖")
#st.markdown("# AI Chat with Docs and Web!👽")
st.markdown(
            """
            <div style='text-align: center;'>
                <h1>Doc-Web AI Chat 💬</h1>
                <p>AI Chat with Docs and Web!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
#st.title("")

st.subheader("Select any chat type👇")
# Create a selectbox with the function names as options
selected_function = st.selectbox("Select a Chat", functions)

if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
else:
    user_api_key = st.sidebar.text_input(
            label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
        )
    if user_api_key:
        st.sidebar.success("OpenAI  API key loaded", icon="🚀")

MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
max_tokens = {"gpt-4":7000, "gpt-4-32k":31000, "gpt-3.5-turbo":3000}
TEMPERATURE_MIN_VALUE = 0.0
TEMPERATURE_MAX_VALUE = 1.0
TEMPERATURE_DEFAULT_VALUE = 0.9
TEMPERATURE_STEP = 0.01
model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
freq_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
temperature = st.sidebar.slider(
            label="Temperature",
            min_value=TEMPERATURE_MIN_VALUE,
            max_value=TEMPERATURE_MAX_VALUE,
            value=TEMPERATURE_DEFAULT_VALUE,
            step=TEMPERATURE_STEP,)

if selected_function == "Chat with Docs":
    main(temperature)
elif selected_function == "Chat with Docs + Web Search":
    st.markdown(
            """
            <div style='text-align: center;'>
                <h4>Enter your <a href="https://serpapi.com/" target="_blank">Serp API key</a> to start conversation with Docs + Web Search</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if os.path.exists(".env") and os.environ.get("SERPAPI_API_KEY") is not None:
            user_serpapi_key = os.environ["SERPAPI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
    else:
        user_serpapi_key = st.sidebar.text_input(
            label="#### Enter SERP API key 👇", placeholder="Paste your SERP API key, sk-", type="password"
        )
        if user_serpapi_key:
            st.sidebar.success("Serp API key loaded", icon="🚀")
    os.environ["OPENAI_API_KEY"] = user_api_key
    doc_search(temperature)
else:
    st.warning("You haven't selected any AI Chat!!")
    
    
