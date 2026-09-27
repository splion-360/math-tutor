# Math Tutor

**Ask a math question. Get a visual lesson made for that question.**

Math is often easier to understand when you can watch an idea unfold. Math Tutor turns a typed question into a short animation with an explanation, narration, and captions.

It is designed for more than elementary math. The same product can explain geometry, calculus, differential equations, Fourier transforms, and other advanced topics.

## From question to lesson

1. The learner asks a question in plain English or with a typed equation.
2. The app estimates the level and chooses a model path.
3. The model writes a Manim animation for that question.
4. The generated code is checked before it is allowed to run.
5. Manim renders the lesson in an isolated container.
6. ElevenLabs provides synchronized narration and captions when available.

![A question moves through routing, generation, validation, rendering, and narration to become a visual lesson.](assets/product-flow.svg)

If a specialist, narration, or media step fails, the app can fall back to a simpler lesson path instead of hiding what happened.

## A tutor that can adapt

A lesson about fractions should not sound like a lecture on partial differential equations. Different questions may need different teaching styles, notation, and visual detail.

The current product uses small LoRA specialists for foundational, intermediate, and advanced lessons. They share one Qwen model, so each specialist only needs to learn a small set of changes.

Our research goes one step further: can the model discover useful specializations during training instead of having us decide them in advance?

That is the idea behind our Dynamic LoRA work. We observe how different training examples try to change the model and look for persistent disagreements that may justify a new specialist.

[Read the plain-language Dynamic LoRA explanation, including the mathematics.](training/src/dynamic_lora/README.md)

## The goal

Math Tutor is not a library of prerecorded clips. Each lesson is generated for the learner's question.

The product goal is simple: make difficult mathematics easier to see, hear, and explore. The research goal is to help the model develop the right kinds of expertise without assuming those categories beforehand.
