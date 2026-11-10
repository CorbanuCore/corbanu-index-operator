// DeepSeek V4.1's three sequential pretokenization rules as typed codepoint scans.
// This function deliberately never calls a regular-expression engine.
std::vector<std::string> unicode_split_deepseek3(const std::string & text, bool byte_encode) {
    const auto cps = unicode_cpts_from_utf8(text);
    using span = std::pair<size_t, size_t>;
    std::vector<span> spans = {{0, cps.size()}};
    auto flags = [&](size_t i) { return unicode_cpt_flags_from_cpt(cps[i]); };
    auto east = [&](size_t i) {
        auto c = cps[i];
        return (c >= 0x4e00 && c <= 0x9fa5) || (c >= 0x3040 && c <= 0x30ff);
    };
    // Isolate numbers (at most three codepoints), then Han/Hiragana/Katakana runs.
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<span> next;
        for (auto s : spans) {
            size_t unmatched = s.first, i = s.first;
            while (i < s.second) {
                const bool match = pass == 0 ? flags(i).is_number : east(i);
                if (!match) { ++i; continue; }
                if (unmatched < i) next.emplace_back(unmatched, i);
                size_t j = i + 1;
                while (j < s.second && (pass == 0 ? (j-i < 3 && flags(j).is_number) : east(j))) ++j;
                next.emplace_back(i,j);
                i = unmatched = j;
            }
            if (unmatched < s.second) next.emplace_back(unmatched,s.second);
        }
        spans = std::move(next);
    }
    std::vector<span> final;
    const std::string punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    auto ascii_letter = [&](size_t i) { auto c=cps[i]; return (c>='A' && c<='Z') || (c>='a' && c<='z'); };
    auto word = [&](size_t i) { auto f=flags(i); return f.is_letter || f.is_accent_mark; };
    auto symbol = [&](size_t i) { auto f=flags(i); return f.is_punctuation || f.is_symbol; };
    auto newline = [&](size_t i) { return cps[i]=='\r' || cps[i]=='\n'; };
    for (auto s : spans) {
        size_t i=s.first, unmatched=s.first;
        while (i<s.second) {
            size_t end=i;
            // ASCII punctuation immediately followed by ASCII letters.
            if (cps[i]<128 && punctuation.find(char(cps[i]))!=std::string::npos && i+1<s.second && ascii_letter(i+1)) {
                end=i+2;
                while (end<s.second && ascii_letter(end)) ++end;
            }
            // Optional non-newline/non-letter/non-punctuation/non-symbol prefix,
            // followed by one or more letters or marks. Backtrack optional prefix.
            if (end==i) {
                auto f=flags(i);
                const bool prefix=!newline(i) && !f.is_letter && !f.is_punctuation && !f.is_symbol;
                size_t begin=(prefix && i+1<s.second && word(i+1)) ? i+1 : i;
                if (word(begin)) {
                    end=begin+1;
                    while (end<s.second && word(end)) ++end;
                }
            }
            // Optional ASCII space followed by punctuation/symbols, then CR/LF.
            if (end==i) {
                size_t begin=(cps[i]==' ' && i+1<s.second && symbol(i+1)) ? i+1 : i;
                if (symbol(begin)) {
                    end=begin+1;
                    while (end<s.second && symbol(end)) ++end;
                    while (end<s.second && newline(end)) ++end;
                }
            }
            if (end==i) {
                size_t ws=i, last_newline=i;
                while (ws<s.second && flags(ws).is_whitespace) {
                    if (newline(ws)) last_newline=ws+1;
                    ++ws;
                }
                if (last_newline>i) end=last_newline;
                else if (ws>i) end=(ws<s.second && ws-i>1) ? ws-1 : ws;
            }
            if (end==i) { ++i; continue; }
            if (unmatched<i) final.emplace_back(unmatched,i);
            final.emplace_back(i,end);
            i=unmatched=end;
        }
        if (unmatched<s.second) final.emplace_back(unmatched,s.second);
    }
    std::vector<std::string> words;
    for (auto s : final) {
        std::string value;
        for (size_t i=s.first;i<s.second;++i) value+=unicode_cpt_to_utf8(cps[i]);
        words.push_back(std::move(value));
    }
    return byte_encode ? unicode_byte_encoding_process(words) : words;
}
