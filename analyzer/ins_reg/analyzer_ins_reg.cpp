#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Instruction {
    int index;
    std::string line;
    std::string mnemonic;
    std::set<int> regs;
};

static std::string trim(const std::string& s) {
    size_t l = 0;
    while (l < s.size() && std::isspace(static_cast<unsigned char>(s[l]))) {
        ++l;
    }

    size_t r = s.size();
    while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
        --r;
    }

    return s.substr(l, r - l);
}

static std::vector<std::string> splitWhitespace(const std::string& s) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

static bool isHexAddressToken(const std::string& s) {
    if (s.empty()) return false;

    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    return true;
}

static std::string extractMnemonic(const std::string& line) {
    auto tokens = splitWhitespace(line);

    /*
        示例：
        00007857 9f285c10      FFMA R39, R16, R125, R39
        tokens:
        [0] 00007857
        [1] 9f285c10
        [2] FFMA
        ...

        也可能有谓词：
        00007857 9f286620  @!P0  CALL.REL.NOINC 0x...
        tokens:
        [0] 00007857
        [1] 9f286620
        [2] @!P0
        [3] CALL.REL.NOINC
    */

    size_t pos = 0;

    // 跳过前两个地址 token
    if (tokens.size() >= 2 && isHexAddressToken(tokens[0]) && isHexAddressToken(tokens[1])) {
        pos = 2;
    }

    // 跳过谓词 token，例如 @P0, @!P0, @P1
    if (pos < tokens.size() && !tokens[pos].empty() && tokens[pos][0] == '@') {
        ++pos;
    }

    if (pos < tokens.size()) {
        return tokens[pos];
    }

    return "";
}

static std::string extractInstructionTextAfterMnemonic(const std::string& line,
                                                       const std::string& mnemonic) {
    size_t p = line.find(mnemonic);
    if (p == std::string::npos) {
        return "";
    }
    return line.substr(p + mnemonic.size());
}

static std::set<int> extractRegistersGeneric(const std::string& text) {
    std::set<int> regs;

    /*
        匹配 R数字，但不匹配：
        - RZ
        - UR数字
        - PR数字等包含在其它标识符里的情况

        (?<![A-Za-z0-9_])R([0-9]+) 在 C++ 标准 regex 里不支持 lookbehind。
        所以这里用捕获前导字符的方式：
        (^|[^A-Za-z0-9_])R([0-9]+)
    */
    std::regex regRe("(^|[^A-Za-z0-9_])R([0-9]+)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), regRe);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        int r = std::stoi((*it)[2].str());
        regs.insert(r);
    }

    return regs;
}

static bool extractFirstOperandRegister(const std::string& operandText, int& reg) {
    /*
        operandText 是助记符之后的字符串，例如：
        " R128, [R145+0x700]"

        对 LDS.128 来说，要找第一个操作数是否为 R数字。
    */
    std::string s = trim(operandText);

    size_t comma = s.find(',');
    std::string first = (comma == std::string::npos) ? s : s.substr(0, comma);
    first = trim(first);

    std::regex firstRegRe("^R([0-9]+)$");
    std::smatch m;
    if (std::regex_match(first, m, firstRegRe)) {
        reg = std::stoi(m[1].str());
        return true;
    }

    return false;
}

static std::set<int> extractRegistersForInstruction(const std::string& mnemonic,
                                                    const std::string& line) {
    std::set<int> regs;

    std::string operandText = extractInstructionTextAfterMnemonic(line, mnemonic);

    // 先普通提取所有 R寄存器
    regs = extractRegistersGeneric(operandText);

    // 特殊处理 LDS.128 目标寄存器
    if (mnemonic == "LDS.128") {
        int base = -1;
        if (extractFirstOperandRegister(operandText, base)) {
            regs.insert(base);
            regs.insert(base + 1);
            regs.insert(base + 2);
            regs.insert(base + 3);
        }
    }

    return regs;
}

static std::string formatRegisterRanges(const std::set<int>& regs) {
    if (regs.empty()) {
        return "(none)";
    }

    std::ostringstream oss;

    auto it = regs.begin();
    int start = *it;
    int prev = *it;
    ++it;

    bool firstRange = true;

    auto emitRange = [&](int a, int b) {
        if (!firstRange) {
            oss << ", ";
        }

        if (a == b) {
            oss << "R" << a;
        } else {
            oss << "R" << a << "-R" << b;
        }

        firstRange = false;
    };

    for (; it != regs.end(); ++it) {
        int cur = *it;
        if (cur == prev + 1) {
            prev = cur;
        } else {
            emitRange(start, prev);
            start = prev = cur;
        }
    }

    emitRange(start, prev);
    return oss.str();
}

static void printSummaryByMnemonic(const std::map<std::string, std::set<int>>& regsByMnemonic) {
    std::cout << "=== Registers used by instruction mnemonic ===\n";

    for (const auto& kv : regsByMnemonic) {
        std::cout << kv.first << " (" << kv.second.size() << ") : "
          << formatRegisterRanges(kv.second) << "\n";
    }

    std::cout << "\n";
}

static void printQueryResult(int reg,
                             const std::vector<int>& instIndices,
                             const std::vector<Instruction>& instructions) {
    if (instIndices.empty()) {
        std::cout << "No instruction uses R" << reg << ".\n";
        return;
    }

    std::cout << "=== Instructions involving R" << reg << " ===\n";

    int prev = -1;
    bool first = true;

    for (int idx : instIndices) {
        if (!first) {
            int skipped = idx - prev - 1;
            if (skipped > 0) {
                std::cout << "... (" << skipped << ")\n";
            }
        }

        const Instruction& ins = instructions[idx];
        std::cout << ins.index << ": " << ins.line << "\n";

        prev = idx;
        first = false;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <sass_file>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream fin(filename);

    if (!fin) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return 1;
    }

    std::vector<Instruction> instructions;
    std::map<std::string, std::set<int>> regsByMnemonic;
    std::unordered_map<int, std::vector<int>> instsByReg;

    std::string line;
    int index = 0;

    while (std::getline(fin, line)) {
        if (trim(line).empty()) {
            continue;
        }

        Instruction ins;
        ins.index = index;
        ins.line = line;
        ins.mnemonic = extractMnemonic(line);

        if (!ins.mnemonic.empty()) {
            ins.regs = extractRegistersForInstruction(ins.mnemonic, line);

            for (int r : ins.regs) {
                regsByMnemonic[ins.mnemonic].insert(r);
                instsByReg[r].push_back(index);
            }
        }

        instructions.push_back(std::move(ins));
        ++index;
    }

    printSummaryByMnemonic(regsByMnemonic);

    std::cout << "Query mode.\n";
    std::cout << "Input register number, for example: 145\n";
    std::cout << "Input q or quit to exit.\n\n";

    while (true) {
        std::cout << "R> ";

        std::string input;
        if (!std::getline(std::cin, input)) {
            break;
        }

        input = trim(input);

        if (input == "q" || input == "quit" || input == "exit") {
            break;
        }

        if (input.empty()) {
            continue;
        }

        // 允许输入 "145" 或 "R145"
        if (!input.empty() && (input[0] == 'R' || input[0] == 'r')) {
            input = input.substr(1);
        }

        bool ok = true;
        for (char c : input) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                ok = false;
                break;
            }
        }

        if (!ok) {
            std::cout << "Invalid input. Please input a register number, such as 145 or R145.\n";
            continue;
        }

        int reg = std::stoi(input);

        auto it = instsByReg.find(reg);
        if (it == instsByReg.end()) {
            static const std::vector<int> empty;
            printQueryResult(reg, empty, instructions);
        } else {
            printQueryResult(reg, it->second, instructions);
        }

        std::cout << "\n\n";
    }

    return 0;
}
