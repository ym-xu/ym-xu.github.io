---
title: "Data Science Tutorials for Clinicians"
date: 2026-11-21
weight: 40
year: "2026"
role: "Author · Teaching"
tags: ["Teaching", "Clinical ML", "Interactive"]
summary: "Interactive courses, in Chinese, that take clinical researchers from their first line of code to prediction models they can validate and report. This page is about how they are designed and taught."
---

A set of interactive tutorials for medical students and clinician-researchers, written in Chinese. I started them for the clinicians I work with on radiotherapy research: people who already collect cohorts, read dose–volume histograms and run survival analyses, and who want to build prediction models on their own data instead of handing the data to someone else.

There are three courses and two practice projects: seven lessons of Python with a six-part practice project, fourteen lessons of statistical machine learning with a five-part practice project on the same data, and an earlier course in R. They are not listed on this site and are shared by link with the people they are written for, so this page describes the design and the teaching and does not link to them. The notes below are about the Python and machine-learning courses and their practice projects, which share one design.

## How it is designed

- **Backwards from the goal.** The destination is a prediction model the reader can build, validate and report. The route is laid out from there: Python for handling clinical data, then regression, cross-validation, calibration, decision curves, sample size and reporting.
- **Forwards from the reader.** Each course is described by where it starts: what the reader already does, such as a baseline table, a Wilcoxon test or a hazard ratio, and what is not required. The Python course opens by turning familiar analyses into code; the machine-learning course moves from "which factors are associated with the outcome" to "how good is the prediction for the next patient".
- **Two formats for two jobs.** Lessons are long-form: explanation, runnable code, short exercises right after each idea. Practice is a separate project on a dataset the lessons do not use, one task per screen, with no explanation and almost no scaffolding: the reader writes the code, gets it checked, and can open two levels of hints before the answer.
- **One clinical thread.** Examples come from the reader's own field, and each course keeps one simulated cohort from beginning to end. The first practice project is a complete analysis, from the dose of a single patient to a written conclusion, built so that a crude comparison and a stratified one disagree and the reader has to find out why. The second returns to the same data with another question, whether the outcome can be predicted, and ends in a short report in which the sample turns out to be too small for more than one predictor and the reader has to say so.
- **No patient data.** Every cohort is simulated with a fixed seed. Code runs in the reader's browser, and the Python course ends on the reader's own computer, so real data never have to leave it.
- **Read on a phone.** Pages, figures and exercises are tested at phone width, because that is where they are read.

## How it teaches

- **Mechanism before name.** Each method is written out as a few concrete steps with a clinical example and coded by hand once; only then comes the library call. The rule is not to explain one term with another.
- **See it, then name it.** Interactive figures let the reader watch resampling, label shuffling, k-fold splits and overfitting before the words appear.
- **Algorithms shown step by step.** Where a method is a procedure (least squares, fitting a logistic curve, ridge and LASSO, growing a tree, a random forest, gradient boosting), a stepped diagram runs it in the page on a small simulated cohort. The reader moves it forward one step at a time; two views change together, the patients and the structure of the model; the numbers above the drawing fill in as it goes; and when a step has played, a slider hands the control to the reader.
- **Worked example, then your turn.** The text works one case and the exercise repeats the analysis on another plan or metric. Every exercise is checked on the spot, and a wrong answer gets a message aimed at that particular mistake.
- **Questions without code.** Each lesson has a few "think about it" questions; every option, right or wrong, explains itself when chosen.
- **Written towards the paper.** Lessons end with how the step reads in an English Methods section and what a reviewer may ask. Terms are given in Chinese with the English beside them.
- **Honest about limits.** Each lesson says what it did not cover, and the machine-learning course says up front which kind of outcome it leaves out.

## How it is built

- A static Hugo site. Python runs in the browser through Pyodide and R through WebR, so there is nothing to install.
- Exercises carry hidden checks; progress is kept in the reader's browser.
- The stepped diagrams compute what they show. The numbers the text quotes are read back from them by the same check that runs the code.
- Before a lesson goes online, a headless browser runs every code block and every reference answer at phone width.
