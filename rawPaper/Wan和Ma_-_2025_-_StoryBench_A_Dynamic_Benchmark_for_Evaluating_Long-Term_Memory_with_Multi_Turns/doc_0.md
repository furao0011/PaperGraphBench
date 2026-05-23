# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns

Luanbo Wan $ ^{1,2,*} $, Weizhi Ma $ ^{1\dagger} $

 $ ^{1} $Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China  

 $ ^{2} $University of Electronic Science and Technology of China, Chengdu, China  

mawz@tsinghua.edu.cn

## Abstract

Long-term memory (LTM) is essential for large language models (LLMs) to achieve autonomous intelligence in complex, evolving environments. Despite increasing efforts in memory-augmented and retrieval-based architectures, there remains a lack of standardized benchmarks to systematically evaluate LLMs' long-term memory abilities. Existing benchmarks still face challenges in evaluating knowledge retention and dynamic sequential reasoning, and in their own flexibility, all of which limit their effectiveness in assessing models' LTM capabilities. To address these gaps, we propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures. These structures simulate real-world scenarios by requiring LLMs to navigate hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions. Our benchmark emphasizes two distinct settings to test reasoning complexity: one with immediate feedback upon incorrect decisions, and the other requiring models to independently trace back and revise earlier choices after failure. As part of this benchmark, we also construct a new dataset designed to test LLMs' LTM within narrative-driven environments. We further validate the effectiveness of our approach through detailed experiments. Experimental results demonstrate the benchmark's ability to robustly and reliably assess LTM in LLMs.

## 1 Introduction

In the field of artificial intelligence, the pursuit of true intelligence in large language models (LLMs) has prompted researchers to look to biology for inspiration [Gutiérrez et al., 2024, Wu et al., 2025]. Just as organisms gradually accumulate knowledge through experience over time, LLMs need to possess long-term memory (LTM) capabilities to achieve self-evolution and strategic optimization in ever-changing environments [Shan et al., 2025]. Moreover, as LLMs are increasingly applied in scenarios such as multi-session dialogue [Zhang et al., 2025], task planning, and lifelong learning, the need for models to retain, update, and leverage prior knowledge dynamically becomes critical. Without robust LTM, AI systems are limited to short-term reasoning and static knowledge use, failing to achieve sustained, autonomous intelligence.

Given the importance of LTM in enabling advanced behaviors, it is crucial to evaluate these capabilities reliably and systematically. However, current benchmarks face challenges in adequately