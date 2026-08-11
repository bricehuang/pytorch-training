# PyTorch training

This is an educational project with the goal of gaining fluency in PyTorch.

I asked ChatGPT to generate 10 PyTorch assignments of increasing difficulty, with accompanying test cases. These ranged from basic tensor algebra (#2) to implementing an autograd-compatible flash attention (#10). 

This repository contains solutions to these 10 assignments in [tasks/](tasks/), and separately a [transformer implementation](transformer/transformer.py) using RoPE and GQA.

## Assignment protocol

Each assignment was implemented twice. In the first implementation, I consulted Google searches and PyTorch documentation, but not coding agents, until all tests passed. Only after all tests passed, I asked Codex to critique my code to make it more idiomatic. The second implementation, in a separate session, was closed-book as much as possible.
