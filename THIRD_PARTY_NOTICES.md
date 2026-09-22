# Third-party notices

## Sho Iizuka — BM25Transformer

The sparse BM25 implementation in this project evolved from earlier project
notebooks and was inspired in part by the computational structure of Sho
Iizuka's [`BM25Transformer`](https://github.com/arosh/BM25Transformer). The
current implementation contains project-specific modifications. This
attribution is retained for transparency and prudence where the complete
pre-Git history of earlier snippets cannot be reconstructed.

`arosh/BM25Transformer` is not an executable dependency of this project. The
positive IDF in this project is algebraically identical to the form documented
by Apache Lucene's `BM25Similarity`; no Lucene source code is incorporated,
and Lucene is not an executable dependency of this project.

The original BSD 3-Clause license text follows verbatim:

```text
BSD 3-Clause License

Copyright (c) 2018, Sho IIZUKA
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
