import validators, streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,WebBaseLoader
from ollama import chat
from langchain_community.chat_models import ChatOllama

## Streamlit App
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website", page_icon="🙂")
st.title("🙂 Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    UseGroq = st.toggle("Use Your Own Groq API")
    if UseGroq:
        groq_api_key=st.text_input("Groq API Key",value="",type="password")
        model=st.text_input("Model",value="")
        llm=ChatGroq(model=model,groq_api_key=groq_api_key)
    else:
        llm = ChatGroq(model='llama-3.3-70b-versatile',api_key="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")


generic_url=st.text_input("URL",label_visibility="collapsed")


prompt_template="""
Provide summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=['text'])


if st.button("Summarize the content from YT or website"):
    if UseGroq and not groq_api_key.strip():
        st.error("Please provide the information to get started")

    elif not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video url or website url")
    
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url)
                else:
                    loader=WebBaseLoader(generic_url)
                
                with st.spinner("Loading documents..."):
                    docs = loader.load()
                    st.write("Documents loaded")

                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)

                with st.spinner("Generating summary..."):
                    output_summary=chain.run(docs)
                    st.success(output_summary)

        except Exception as e:

            st.exception(f"Exception:{e}")


