"""Apply the audited local Linux/tokenizer patch to the pinned JigSaw source."""
import argparse
from pathlib import Path
import subprocess


def replace(path, old, new):
    text = path.read_text()
    if new in text: return
    if text.count(old) != 1: raise ValueError('Unexpected upstream source: '+str(path))
    path.write_text(text.replace(old,new))


def patch(root):
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if revision!='3c95ac82c19e496348fe57d06f083880a7d86e95': raise ValueError('Wrong upstream revision')
    replace(root/'src/llama-moe-stream.cpp','#include <cinttypes>','#include <cinttypes>\n#include <cmath>')
    replace(root/'src/unicode.h','size_t unicode_len_utf8(char src);',
            'std::vector<std::string> unicode_split_deepseek3(const std::string & text, bool byte_encode = true);\n\nsize_t unicode_len_utf8(char src);')
    source=(Path(__file__).parent/'deepseek3_split.cpp.inc').read_text()
    replace(root/'src/unicode.cpp','std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode) {',
            source+'\n\nstd::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode) {')
    route='const auto word_collection = (vocab.get_pre_type() == LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM\n            || vocab.get_pre_type() == LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM\n            || vocab.get_pre_type() == LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE)\n            ? unicode_split_deepseek3(text, tokenizer.byte_encode)\n            : unicode_regex_split(text, tokenizer.regex_exprs, tokenizer.byte_encode);'
    previous='const auto word_collection = vocab.get_pre_type() == LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM\n            ? unicode_split_deepseek3(text, tokenizer.byte_encode)\n            : unicode_regex_split(text, tokenizer.regex_exprs, tokenizer.byte_encode);'
    old=previous if previous in (root/'src/llama-vocab.cpp').read_text() else 'const auto word_collection = unicode_regex_split(text, tokenizer.regex_exprs, tokenizer.byte_encode);'
    replace(root/'src/llama-vocab.cpp',old,route)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path)
    patch(parser.parse_args().source)
