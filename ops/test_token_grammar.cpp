#include "llama.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

std::vector<llama_token> encode(const llama_vocab * vocab,const std::string & text) {
    int n=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
    std::vector<llama_token> ids(-n);
    n=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),false,true);
    if(n<0)throw std::runtime_error("Tokenizer failed");
    ids.resize(n);return ids;
}

bool allowed(llama_sampler * grammar,llama_token id) {
    llama_token_data item={id,0.0f,0.0f};
    llama_token_data_array array={&item,1,-1,false};
    llama_sampler_apply(grammar,&array);
    return std::isfinite(item.logit);
}

int main(int argc,char **argv) {
    if(argc!=3)return 2;
    auto params=llama_model_default_params();params.vocab_only=true;
    auto * model=llama_model_load_from_file(argv[1],params);
    if(!model)return 3;
    const auto * vocab=llama_model_get_vocab(model);
    std::ifstream file(argv[2]);
    std::string final_grammar((std::istreambuf_iterator<char>(file)),{});
    std::string thinking_grammar=final_grammar;
    const std::string old="root ::= ws object ws";
    if(thinking_grammar.find(old)!=0)return 4;
    thinking_grammar.replace(0,old.size(),"root ::= \"</think>\" ws object ws");
    auto trigger=encode(vocab,"</think>");
    if(trigger.size()!=1)return 5;
    auto * lazy=llama_sampler_init_grammar_lazy_patterns(vocab,thinking_grammar.c_str(),"root",nullptr,0,trigger.data(),1);
    auto * direct=llama_sampler_init_grammar(vocab,final_grammar.c_str(),"root");
    if(!lazy || !direct)return 6;
    const auto word=encode(vocab,"Hello").at(0);
    if(!allowed(lazy,word) || allowed(direct,word))return 7;
    for(auto id:encode(vocab,"Reasoning before JSON. ")) {
        if(!allowed(lazy,id))return 8;
        llama_sampler_accept(lazy,id);
    }
    llama_sampler_accept(lazy,trigger[0]);
    if(allowed(lazy,word))return 9;
    for(auto id:encode(vocab," \n{\"score\":70,\"brief\":\"valid syntax\"}\r\n")) {
        if(!allowed(lazy,id) || !allowed(direct,id))return 10;
        llama_sampler_accept(lazy,id);llama_sampler_accept(direct,id);
    }
    if(!allowed(lazy,llama_vocab_eos(vocab)) || !allowed(direct,llama_vocab_eos(vocab)))return 11;
    llama_sampler_free(lazy);llama_sampler_free(direct);llama_model_free(model);
    std::cout<<"Token-triggered thinking and immediate finalizer JSON grammars passed\n";
}
