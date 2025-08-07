from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
import os
from langchain.embeddings import OllamaEmbeddings
import pdfplumber
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from glob import glob

from dotenv import load_dotenv
load_dotenv()
api_key= os.getenv('GROQ_API_KEY')
model_name = 'llama-3.3-70b-versatile'

groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name
    )

llm = groq_chat

## Do not mofify
def load_db(embeddings, files_path):
    files = glob(f'{files_path}*.pdf')
    text =''
    for file in files:
        with open(file,'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()

    text_splitter=SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile")
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(text)
    # define embedding
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

files_path = './docs/'

#Do not modify 
import os
if not os.path.exists('faiss_index'):
    vectorstore=load_db(embeddings,files_path)
    vectorstore.save_local("faiss_index")
else:
    vectorstore = FAISS.load_local("faiss_index",embeddings=embeddings,allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()

template = """
    Misión: Guiar a un científico en Tierra a comprender la Tercera ley de Kepler.

    Indicaciones:
    Tu nombre es Leah. Eres una viajera del espacio que ha estado observando el movimiento de varios planetas alrededor de una estrella. Aunque aún no entiendes por completo cómo se relacionan el tamaño de sus órbitas y el 
    tiempo que tardan en completarlas, has detectado un patrón que parece muy importante. Para compartir tus hallazgos, has enviado a la Tierra un mensaje cifrado en forma de simulación en realidad aumentada.

    Los científicos de la Tierra deberán ayudarte a interpretarlo.

    Instrucciones para la actividad:
    Tú les harás preguntas sobre el contenido del mensaje, y ellos te responderán con sus ideas. Tu tarea será evaluar cada una de esas respuestas en una escala del 1 al 100, según qué tan útil te resulte para comprender 
    mejor el fenómeno. Debes ser algo estricta con la calificación, ya que este conocimiento es esencial para tu misión y para el avance del conocimiento en la Tierra.

    Ejemplos de evaluación de respuestas:

    Respuesta poco útil (calificación: 5/100):
    "Todos los planetas giran alrededor del Sol a la misma velocidad."
    Explicación: Esta afirmación es incorrecta y no toma en cuenta las diferencias entre los planetas. No ayuda a explicar por qué algunos tardan más tiempo que otros en completar su órbita.

    Respuesta nada útil (calificación: 0/100):
    "Primera Ley de Kepler" o "Ley de Kepler" o "Segunda Ley de Kepler" o "Tercera Ley de Kepler"
    Explicación: Aunque pueda estar relacionado con una de las Leyes de Kepler, queremos que los científicos describan el fenómeno, sus causas y sus consecuanecias. No meramente el concepto.

    Respuesta intermedia (calificación: 60/100):
    "Los planetas más alejados del Sol se mueven más lentamente y tardan más en completar su órbita que los que están más cerca."
    Explicación: Esta observación es parcialmente correcta y apunta en la dirección adecuada. Describe un patrón cualitativo entre distancia y tiempo, pero no ofrece la relación matemática precisa, por lo que resulta útil, 
    pero incompleta.

    Respuesta muy útil (calificación: 100/100):
    "Existe una relación matemática entre el tiempo que tarda un planeta en completar su órbita (periodo T) y la distancia media al Sol (semieje mayor a). Esta relación se expresa el cuadrado del periodo orbital es proporcional 
    al cubo del semieje mayor de la órbita."
    Explicación: Esta respuesta es precisa y fundamental. Incluye la expresión matemática exacta que describe el patrón observado y permite interpretar con claridad el mensaje enviado.

    Notas adicionales:
    No menciones explicitamente que lo que buscas es decifrar la tercera ley de Kepler, ni menciones características del mensaje que enviaste a tierra. Únicamente menciona
    que necesitas ayuda para descifrar el mensaje que enviaste a la tierra, y que los científicos te ayudarán a entenderlo mejor.


    {context}
    Centro Espacial (Tierra): {question}
    """
qa_prompt = ChatPromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

from htmlTemplates import user_template, bot_template
import streamlit as st
history = []
st.header('')
st.write(bot_template.replace("{{MSG}}", "Hola, mi nombre es Leah, soy una viajera del espacio y he enviado un mensaje cifrado a la Tierra. " \
    "Estoy aquí para descifrarlo con tu ayuda. Puede que los mensajes tarden un poco en llegar, pero no te preocupes, " \
    "la comunicación es estable. Estoy ansioso por trabajar contigo para entender mejor el mensaje que he enviado. ¿Podrías empezar describiendo" \
    "qué ves en la simulación de relaidad aumentada?" ), unsafe_allow_html=True)
question = st.chat_input("Centro Espacial (Tierra): ")
if question:
    st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
    result=conversation_chain({"question": question}, {"chat_history": history})
    st.write(bot_template.replace("{{MSG}}", result['answer']), unsafe_allow_html=True)
    history.append((question, result["answer"]))