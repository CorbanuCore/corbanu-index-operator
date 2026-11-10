#include "unicode.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

int main() {
    using words=std::vector<std::string>;
    std::vector<std::pair<std::string,words>> cases={
        {"",{}},{"1234567",{"123","456","7"}},
        {"a1234b",{"a","123","4","b"}},
        {"Hello, world!",{"Hello",","," world","!"}},
        {"'helloÉ",{"'hello","É"}},
        {"a中日あアb",{"a","中日あア","b"}},
        {"  hello",{" "," hello"}},
        {"x   ",{"x","   "}},
        {" \t\r\n  x",{" \t\r\n"," "," x"}},
        {"\t\t",{"\t\t"}},
        {"🐋!\r\n",{"🐋!\r\n"}},
        {"a\u0301b",{"a\u0301b"}},
        {"\u0001\u0002A",{"\u0001","\u0002A"}},
        {"१२३४",{"१२३","४"}},
        {"a\u00a0b",{"a","\u00a0b"}},
        {"a12.34",{"a","12",".","34"}},
        {" <tag>",{" <","tag",">"}}
    };
    for (auto & item:cases) {
        if (unicode_split_deepseek3(item.first,false)!=item.second) {
            std::cerr<<"Segmentation failed for: "<<item.first<<"\n";
            return 1;
        }
    }
    std::mt19937 generator(438916795);
    for (size_t sample=0;sample<10000;++sample) {
        std::string text;
        for (size_t i=0;i<50;++i) {
            uint32_t cp=generator()%0x110000;
            if (cp>=0xd800 && cp<=0xdfff) cp='a';
            text+=unicode_cpt_to_utf8(cp);
        }
        std::string joined;
        for (auto & word:unicode_split_deepseek3(text,false)) joined+=word;
        if (joined!=text) throw std::runtime_error("Unicode data lost");
    }
    std::cout<<"17 segmentation cases and 10000 Unicode preservation samples passed\n";
}
