---
title: "Data Science Tutorials for Clinicians"
date: 2026-11-21
weight: 40
year: "2026"
role: "Author · Teaching"
tags: ["Teaching", "Clinical ML", "Interactive"]
summary: "Interactive courses, in Chinese, that take clinical researchers from their first line of code to prediction models they can validate and report. Code runs and is checked in the browser."
---

A set of interactive tutorials for medical students and clinician-researchers, written in Chinese. I started them for the clinicians I work with on radiotherapy research: people who already collect cohorts, read dose–volume histograms and run survival analyses, and who want to build prediction models on their own data instead of handing the data to someone else.

## What is in it

- **[Python for medical students](/python/)**: seven lessons, from the first line of code to a baseline table, figures for a paper and the common tests. It starts from analyses the reader already runs in R or SPSS and turns them into code.
- **[A themed practice](/python/practice/)** for the Python course: an audit of brachytherapy dosimetry in cervical cancer on a simulated department dataset. Six parts, one task per screen, from the EQD2 of a single patient to a written audit conclusion.
- **[Statistical machine learning for medical students](/stat-ml/)**: fourteen lessons planned, from regression models to cross-validation, calibration, decision curves, sample size and reporting with TRIPOD+AI. The thread is the step from "which factors are associated with the outcome" to "how good is the prediction for the next patient". In progress; the lesson on cross-validation is online.
- **[R for Cancer Research](/r-course/)**: thirteen lessons, from setup and data wrangling to survival analysis, competing risks and EQD2.

## How it teaches

- **Start from what the reader already does.** A baseline table, a Wilcoxon test, a hazard ratio: familiar analyses come first, as code; what is new is added on top.
- **Mechanism before name.** Each method is written out as a few concrete steps with a clinical example and coded by hand once; only then comes the library call. Interactive figures let the reader watch resampling, label shuffling, k-fold splits and overfitting before the terms appear.
- **Worked example, then your turn.** The text works one case and the exercise repeats the analysis on another plan or metric. Every exercise is checked on the spot, and a wrong answer gets a message aimed at that particular mistake.
- **Written towards the paper.** Lessons in the Python and machine-learning courses end with how the step reads in an English Methods section and what a reviewer may ask.

## How it is built

- A static Hugo site. Python runs in the browser through Pyodide and R through WebR, so there is nothing to install and the reader's code stays on their own device.
- Exercises carry hidden checks; progress is kept in the browser.
- In the Python and machine-learning courses every cohort is simulated with a fixed seed, in the shape of real brachytherapy data. The only real numbers are quoted from a published paper, with the reference.
- Before a lesson goes online, a headless browser runs every code block and every reference answer at phone width.
