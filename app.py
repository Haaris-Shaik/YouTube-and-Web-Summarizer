import validators, streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,WebBaseLoader

## Streamlit App
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website", page_icon="🙂")
st.title("🙂 Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

prompt_template="""
Provide summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=['text'])


if st.button("Summarize the content from YT or website"):
    if not groq_api_key.strip() or not generic_url.strip():
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