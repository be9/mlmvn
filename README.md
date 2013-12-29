mlmvn
=====

This is MLMVN code based on [Igor Aizenberg book](http://www.freewebs.com/igora/CVNN-MVN_book.htm).

Currently it contains implementation of both MVN and MLMVN learning. Two examples from book are implemented:

* Post function example (Chapter 3, page 129).
* "Three classes" example with 2-2-1 MLMVN (Chapter 4, page 164).

To build the library and examples you need a decent C++ compiler (I used both clang provided by Apple and gcc 4.6.2) and CMake build tool.

Roadmap
-------

* Implement classifier framework with rejection sectors and "winner" detection.
* Add OpenMP support to utilize multiple CPU cores.
* Implement UBN and MVN-P.

Pull requests are welcome.
