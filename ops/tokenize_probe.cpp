#include "llama.h"
#include "json.hpp"
#include <iostream>
#include <vector>

int main(int argc,char **argv) {
    if (argc!=2) return 2;
    auto params=llama_model_default_params(); params.vocab_only=true;
    auto * model=llama_model_load_from_file(argv[1],params);
    if (!model) return 3;
    const auto * vocab=llama_model_get_vocab(model);
    std::string line;
    while (std::getline(std::cin,line)) {
        auto input=nlohmann::json::parse(line);
        std::string text=input.at("text");
        int count=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
        std::vector<llama_token> ids(-count);
        count=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),false,true);
        if (count<0) return 4;
        ids.resize(count);
        std::string decoded;
        for (auto id:ids) {
            std::vector<char> piece(256);
            int n=llama_token_to_piece(vocab,id,piece.data(),piece.size(),0,true);
            if (n<0) {piece.resize(-n);n=llama_token_to_piece(vocab,id,piece.data(),piece.size(),0,true);}
            if (n<0) return 5;
            decoded.append(piece.data(),n);
        }
        std::cout<<nlohmann::json({{"ids",ids},{"decoded",decoded}}).dump()<<"\n";
    }
    llama_model_free(model);
}
