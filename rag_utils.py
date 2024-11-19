from typing import List, Dict
from dataclasses import dataclass
from langchain.agents import AgentType
from pydantic import BaseModel, validator, Extra
from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
import re
import asyncio
from rank_bm25 import BM25Okapi
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage, 
    LLMResult
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Class definitions
@dataclass
class Llm():
    llm: BaseChatModel
    llm_name: str
    llm_args: dict
        
    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f"llm: {self.llm_name} \n llm_args: {self.llm_args}"

class OpenAI(ChatOpenAI):
    # TODO: change model_name and 
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    openai_api_key: str
    streaming: bool = True

    @staticmethod
    def get_display_name():
        return "OpenAI"

    @staticmethod
    def get_valid_model_names():
        valid_model_names = {"gpt-4o-mini","gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", 
                            "gpt-3.5-turbo-16k-0613", "gpt-4", "gpt-4-0613", "gpt-4-32k-0613", "gpt-4-32k"}
        return valid_model_names

    @validator("model_name")
    def validate_model_name(cls, request):
        valid_model_names = cls.get_valid_model_names()
        if request not in valid_model_names:
            raise ValueError(f"invalid model name given - {request} , valid ones are {valid_model_names}")
        return request

class LangchainLlms:
    def __init__(self):
        self.__llms = {
            "OpenAI": {
                "llm": OpenAI,
                "schema": OpenAI
            }
        }

    def get_llm(self, llm_name: str, **llm_kwargs) -> Llm:
        if llm_name not in self.__llms:
            raise ValueError(f"invalid llm name given - {llm_name} , must be one of {list(self.__llms.keys())}")
        llm = self.__llms[llm_name]["llm"]
        llm_args = self.__llms[llm_name]["schema"](**llm_kwargs)
        llm_obj = llm(**dict(llm_args))
        return Llm(llm=llm_obj,llm_args=dict(llm_args), llm_name=llm_name)

class VectorDbWithBM25:
    def __init__(self, vector_db, bm25_corpus):
        self.__vector_db = vector_db
        self.__bm25_corpus = bm25_corpus    
        tokenized_corpus = [doc.split(" ") for doc in bm25_corpus]
        self.__bm25 = BM25Okapi(tokenized_corpus)
        
    def vector_db_search(self, query: str, k=3) -> Dict[str, float]:
        search_result = dict()
        docs_and_scores = self.__vector_db.similarity_search_with_relevance_scores(query=query, k=k)
        for doc, score in docs_and_scores:
            search_result[doc.page_content] = score
            
        return {doc: score for doc, score in sorted(search_result.items(), key=lambda x: x[1], reverse=True)}
    
    def bm25_search(self, query: str, k=3) -> Dict[str, float]:
        tokenized_query = query.split(" ")
        doc_scores = self.__bm25.get_scores(tokenized_query)
        docs_with_scores = dict(zip(self.__bm25_corpus, doc_scores))
        sorted_docs_with_scores = sorted(docs_with_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_docs_with_scores[:k])
    
    def combine_results(self, vector_db_search_results: Dict[str, float], 
                            bm25_search_results: Dict[str, float]) -> Dict[str, float]:
            #combine vector_search and bm25 search query result into a dictionary with query as key and normalized score as value
            def normalize_dict(input_dict):
                epsilon = 0.05
                min_value = min(input_dict.values())
                max_value = max(input_dict.values())
                a, b = 0.05, 1
                
                if max_value == min_value:
                    return {k: b if max_value > 0.5 else a for k in input_dict.keys()}
        
                return {k: a + ((v - min_value) / (max_value - min_value)) * (b - a) for k, v in input_dict.items()}
            
            norm_vector_db_search_results = normalize_dict(vector_db_search_results)
            norm_bm25_search_results = normalize_dict(bm25_search_results)
    
            # Combine the dictionaries
            combined_dict = {}
            for k, v in norm_vector_db_search_results.items():
                combined_dict[k] = v
    
            for k, v in norm_bm25_search_results.items():
                if k in combined_dict:
                    combined_dict[k] = max(combined_dict[k], v)
                else:
                    combined_dict[k] = v
    
            return combined_dict
    
    # combine bm25_search with vector search if possible, combine the result based on rank.
    def search(self, query: str, k=3, do_bm25_search=True) -> Dict[str, float]:
        vector_db_search_results = self.vector_db_search(query, k=k)
        
        if do_bm25_search:
            bm25_search_results = self.bm25_search(query, k=k)
            if bm25_search_results:
                combined_search_results = self.combine_results(vector_db_search_results, bm25_search_results)
                sorted_docs_with_scores = sorted(combined_search_results.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_docs_with_scores)
        return vector_db_search_results


def remove_bullet_points(text):
    lines = text.strip().split('\n')
    cleaned_lines = [re.sub(r'^[\d\.\-\*\s]+', '', line).strip() for line in lines]
    return cleaned_lines

class RagFusion:
    def __init__(self, vector_store, llm):
        self.__vectorstore = vector_store
        self.__llm = llm
        
    async def generate_queries(self, query: str) -> List[str]:
        system_prompt = "You are a helpful assistant that generates multiple search queries based on a single input query."
        human_message = f"Generate 4 search queries related to: {query}"
        messages = []
        messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=human_message))
        
        response = await self.__llm.agenerate(messages=[messages])
        if response and isinstance(response, LLMResult):
            generations = response.flatten()
            llm_result = generations[0].generations[0][0].text
            parsed_result = remove_bullet_points(llm_result)
            
            return parsed_result
        
        return []
    
    async def rewrite_query(self, query: str) -> str:
        
        prompt = f"""Provide a better search query for web search engine to answer the given question. End \
the queries with ’**’. Question:  ``` {query} ``` """
        
        messages = [HumanMessage(content=prompt)]
        response = await self.__llm.agenerate(messages=[messages])
        if response and isinstance(response, LLMResult):
            generations = response.flatten()
            llm_result = generations[0].generations[0][0].text
            return llm_result.strip("**")
        
        return ""
        
        
    def vector_db_search(self, query: str, k=3) -> Dict[str, float]:
        search_result = dict()
        docs_and_scores = self.__vectorstore.search(query, do_bm25_search=True, k=k)
        for doc, score in docs_and_scores.items():
            search_result[doc] = score
            
        return {doc: score for doc, score in sorted(search_result.items(), key=lambda x: x[1], reverse=True)}

            
        
    def retrieve_multiple_responses(self, similar_queries: List[str], k=3) -> Dict[str, Dict[str, float]]:
        all_results = dict()
        for query in similar_queries:
            search_results = self.vector_db_search(query, k=k)
            all_results[query] = search_results
        
        return all_results
    
    def reciprocal_rank_fusion(self, search_results_dict, k=60) -> Dict[str, float]:
        # k=60 is taken from the paper https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        
        fused_scores = {}
        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                previous_score = fused_scores[doc]
                fused_scores[doc] += 1 / (rank + k)

        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        return reranked_results
    
    async def run_spr(self, content_to_compress: str) -> str:
        spr_system_prompt = """# MISSION
You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.

# THEORY
LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of a LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

# METHODOLOGY
Render the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human."""
        
        human_message = f"this is the input content that you need to distill - ``` {content_to_compress} ``` "
        
        messages = []
        messages.append(SystemMessage(content=spr_system_prompt))
        messages.append(HumanMessage(content=human_message))
        
        response = await self.__llm.agenerate(messages=[messages])
        if response and isinstance(response, LLMResult):
            generations = response.flatten()
            llm_result = generations[0].generations[0][0].text
            return llm_result
        
        return ""
    
    async def form_final_result(self, spr_results: List[str], original_query: str) -> str:
        spr_results = "\n ****************** \n".join(spr_results)
        
        prompt = f"""Answer the user's question based only on the following context:

                    <context>
                    {spr_results}
                    </context>

                    Question: ``` {original_query} ``` 
                    Try adhere to the template as much as possible. """
        
        messages = [HumanMessage(content=prompt)]
        response = await self.__llm.agenerate(messages=[messages])
        if response and isinstance(response, LLMResult):
            generations = response.flatten()
            llm_result = generations[0].generations[0][0].text
            return llm_result
        
        fallback_prompt = f"""Answer the following question as best as you can without any additional context.

                            Question: ``` {original_query} ``` """
        fallback_messages = [HumanMessage(content=fallback_prompt)]
        fallback_response = await self.__llm.agenerate(messages=[fallback_messages])
        if fallback_response and isinstance(fallback_response, LLMResult):
            fallback_generations = fallback_response.flatten()
            fallback_llm_result = fallback_generations[0].generations[0][0].text
            return fallback_llm_result
        
        return "No response could be generated."
        
    async def arun(self, query: str, rewrite_original_query=False):
        
        if rewrite_original_query:
            rephrased_query = await self.rewrite_query(query)
            if rephrased_query:
                query = rephrased_query
                print("rephrased_query: ", rephrased_query)
                print()
            
        similar_queries_list = await self.generate_queries(query)
        print("similar_queries_list: ", similar_queries_list)
        print()
        if similar_queries_list:
            search_results = self.retrieve_multiple_responses(similar_queries_list)
            reranked_results = self.reciprocal_rank_fusion(search_results)
            
            # here I am using all the reranked results, you can select the top N
            spr_tasks = []
            spr_results = []
            
            for result, score in reranked_results.items():
                spr_task = asyncio.create_task(self.run_spr(result))
                spr_tasks.append(spr_task)
                
            done, pending = await asyncio.wait(spr_tasks, timeout=180)
            for done_task in done:
                if done_task.exception() is None:
                    result = done_task.result()
                    spr_results.append(result)
                    
            for pending_task in pending:
                pending_task.cancel()
            
            if spr_results:
                
                for spr_content in spr_results:
                    print(spr_content)
                    print()
                
                print("*" * 100)
                
                final_result = await self.form_final_result(spr_results, query)
                print("final result: ")
                return (final_result)