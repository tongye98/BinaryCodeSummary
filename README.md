# Binary Code Summarization
Official implementation of EMNLP 2023 main conference paper "CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code".

`Note! We will soon upload relevant training and usage instructions.`

## Binary Projects

The list of 51 binary projects and their corresponding versions utilized for constructing the dataset.

| Binary Projects | Version | Binary Projects | Version |
| --- | --- | --- | --- |
| a2ps | 4.14 | binutils | 2.30 |
| bool | 0.2.2 | ccd2cue | 0.5 |
| cflow | 1.5 | coreutils | 8.29 |
| cpio | 2.12 | cppi | 1.18 |
| dap | 3.10 | datamash | 1.3 |
| direvent | 5.1 | enscript | 1.6.6 |
| findutils | 4.6.0 | gawk | 4.2.1 |
| gcal | 4.1 | gdbm | 1.15 |
| glpk | 4.65 | gmp | 6.1.2 |
| gnudos | 1.11.4 | grep | 3.1 |
| gsasl | 1.8.0 | gsl | 2.5 |
| gss | 1.0.3 | gzip | 1.9 |
| hello | 2.10 | inetutils | 1.9.4 |
| libiconv | 1.15 | libidn2 | 2.0.5 |
| libmicrohttpd | 0.9.59 | libosip2 | 5.0.0 |
| libtasn1 | 4.13 | llibtool | 2.4.6 |
| libunistring | 0.9.10 | lightning | 2.1.2 |
| macchanger | 1.6.0 | nettle | 3.4 |
| patch | 2.7.6 | plotutils | 2.6 |
| readline | 7.0 | recutils | 1.7 |
| sed | 4.5 | sharutils | 4.15.2 |
| spell | 1.1 | tar | 1.30 |
| texinof | 6.5 | time | 1.9 |
| units | 2.16 | vmlinux | 4.1.52 |
| wdiff | 1.2.2 | which | 2.21 |
| xorriso | 1.4.8 | - | - |

## Citation
```
@inproceedings{ye-etal-2023-cp,
    title = "{CP}-{BCS}: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code",
    author = "Ye, Tong  and
      Wu, Lingfei  and
      Ma, Tengfei  and
      Zhang, Xuhong  and
      Du, Yangkai  and
      Liu, Peiyu  and
      Ji, Shouling  and
      Wang, Wenhai",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.911",
    doi = "10.18653/v1/2023.emnlp-main.911",
    pages = "14740--14752",
    abstract = "Automatically generating function summaries for binaries is an extremely valuable but challenging task, since it involves translating the execution behavior and semantics of the low-level language (assembly code) into human-readable natural language. However, most current works on understanding assembly code are oriented towards generating function names, which involve numerous abbreviations that make them still confusing. To bridge this gap, we focus on generating complete summaries for binary functions, especially for stripped binary (no symbol table and debug information in reality). To fully exploit the semantics of assembly code, we present a control flow graph and pseudo code guided binary code summarization framework called CP-BCS. CP-BCS utilizes a bidirectional instruction-level control flow graph and pseudo code that incorporates expert knowledge to learn the comprehensive binary function execution behavior and logic semantics. We evaluate CP-BCS on 3 different binary optimization levels (O1, O2, and O3) for 3 different computer architectures (X86, X64, and ARM). The evaluation results demonstrate CP-BCS is superior and significantly improves the efficiency of reverse engineering.",
}
```