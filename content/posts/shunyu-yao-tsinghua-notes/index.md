---
title: "Notes on Shunyu Yao's talk at Tsinghua"
date: 2026-11-21
slug: "shunyu-yao-tsinghua-notes"
summary: "Shunyu Yao spoke at Tsinghua on 18 November 2026 in a closed session. What the published reports say he argued, restated in my own words, with his remarks in Hong Kong earlier in the month, and my view on each part from my work on document agents."
tags: ["AI", "Agents"]
---

On 18 November 2026, Shunyu Yao (姚顺雨) went back to the Institute for Interdisciplinary Information Sciences at Tsinghua, where he studied in the Yao Class, for a two-hour closed conversation with his former classmate Kaifeng Lyu (吕凯风), now an assistant professor there [[2]][s2]. Yao wrote ReAct, co-wrote SWE-bench and τ-bench, worked on Operator and Deep Research at OpenAI, and now leads Tencent's Hunyuan models as its chief AI scientist [[2]][s2]. I pay attention to what he says because he has built agents, built benchmarks for them, and now trains the models underneath. My own work, on agents that read long documents, touches all three.

There is no recording and no transcript. Two write-ups circulated afterwards: notes that the blogger 幻觉工程师 made from materials of the talk [[1]][s1], and a ten-point summary by the WeChat account 十字路口Crossing, based on the full conversation [[2]][s2]. Below I restate what they report in my own words, without quoting, and give my view after each part. Where the two disagree I say so. They disagree even on the day: the notes, reposted on the 20th, call it yesterday. The numbers in brackets link to the sources, which are listed at the end.

## Small experiments, small models

After ReAct, Yao tried to post-train LLaMA 13B on agent trajectories. The results were bad enough that he began to doubt himself as a researcher. Only later, with OpenAI's compute, did he conclude that the model had been too small for the idea to show [[1]][s1] [[2]][s2]. The lesson he drew is that with little compute, noise can hide a real effect and mislead you about which problems are worth pursuing. Small experiments, together with theory and scaling laws, are for predicting the large run, and what works on a small model does not necessarily carry over to a large one [[1]][s1] [[2]][s2].

He also expects small models to matter more, not less. Once AI is used to do AI research, small models are cheap enough for many experiments. He pictures one strong model distilled into a hierarchy of sizes, each matched to the cost of its task [[1]][s1]. And he called recursive self-improvement a plain idea, using AI to run the research loop better, where progress has always come from the people who fix bugs, clean data and run experiments rather than from those who coin new terms [[1]][s1].

**My view.** The two points fit together if small models are used to predict, not to decide. A university lab lives at the small end, so this is my normal situation, and [AdaNav](/publications/icdar-adanav/) gave me a version of his story. With the same backbone as the baselines, Qwen2.5-VL-32B, it only tied the best of them on MMLongBench-Doc, 59.5 against 59.6. With Qwen3-VL-30B-A3B it reached 65.7, although the baselines were not rerun on that model. Had we stopped at the first backbone, we would have reported a rough tie with a smaller index. A result about navigation is a result about the navigator and the backbone together, and a negative result should name both, with their size.

His hierarchy also describes what can happen inside one pipeline. In AdaNav, a regular expression catches explicit references such as "Table 3", which are then looked up without any model call; a single LLM call routes the other questions; and the reasoner reads only the pages chosen for it, about four per question. Cost can be matched to each question as well as to each task.

## Reinforcement learning: few assumptions, correctness first

Post-training at scale has come back to plain policy gradient and close variants such as GRPO, he said, because they assume the least. More elaborate methods looked good in small experiments, and each extra assumption was one more way to fail on a large cluster [[1]][s1] [[2]][s2]. For RL engineering his order is correctness, then stability, then efficiency, and the training and inference systems have to be designed together [[1]][s1].

For post-training data he separated two aims. Ever harder math problems build reasoning. Tasks such as writing are computationally easier and carry much of the real use, and a model can be strong at math and still weak at the work users bring to it [[1]][s1]. He treats many alignment problems as reward design: a child rewarded only for full marks in math will put everything into math, and a model does the same with its reward. He sees this as an engineering problem, to be improved through data, rewards and evaluation [[1]][s1].

**My view.** This is the part closest to what I want to work on next, reinforcement learning for agents that work over documents. There the reward is usually a judge's verdict on the final answer, and the judge is an assumption like any other. Correctness first means checking the judge before handing it to an optimizer. In [MADQA](/publications/icml-madqa/) the judge was calibrated against human judgments before we used it for evaluation, and a reward deserves at least as much.

Designing training and inference together also reaches further for agents. A rollout over a long document is a sequence of searches, page renders and model calls, so the environment belongs to the system as much as the inference server does, and its speed decides how many rollouts you can afford.

His reward analogy has a document version. If the reward looks only at the final answer, nothing in it tells an agent that one more rephrasing of a failed query is wasted. MADQA found agents doing exactly this, while people changed strategy quickly once a first query failed. Some measure of effort, like the one MADQA uses for evaluation, belongs in the reward too.

## Evaluation

Both write-ups have him criticizing metric chasing. When the people a team reports to do not know the technology well, the team reports what is easy to count: parameters, data volume, leaderboard points, hardware utilization. Doubling utilization makes a good report, while two weeks spent fixing a bug in the correctness of the computation makes none [[1]][s1] [[2]][s2]. An evaluation exists to show you your weak points, and once you tune to it, it stops doing that. In the Hunyuan team, according to [[2]][s2], a core benchmark is run once and then sealed. His check against fooling yourself: if a model matches GPT on leaderboards but earns a hundredth or a thousandth of its revenue, the problem is one the leaderboard does not show [[1]][s1] [[2]][s2].

**My view.** As a co-author of a benchmark, I read this from the other side. The authors of a public benchmark cannot seal it, and once it is out, people will tune to it. What they can do is make a single number insufficient. MADQA chose its test questions for how well they separate strong systems from weak ones, and it scores the pages an agent cites and the effort it spends, not only the answer. The main result shows why. People and the best agent both reach 82.2%, yet they solve different questions (Cohen's κ = 0.24), and the agent needs about five times the effort. One accuracy figure would have called them equal.

The sealing rule is easy to adopt in a small lab: a test set that is opened once, at the end.

## Data

He does not think the competition over pretraining data is over. Good data such as code is expensive, so it gets reused, and reuse brings memorization; synthetic data can lose quality; how to build data dynamically and how to mix sources are still open questions [[1]][s1]. He sees no hard limit on public data in the next year or two [[2]][s2]. The way data is produced changes every few years, from paying people to draw bounding boxes, to labeling with small models, to large models making their own data and playing against themselves. What lasts is a team's taste for what good data is [[2]][s2].

He also pointed at the large amount of industrial data that is not text: time series, clinical records, tables. Models trained natively on these modalities often reason poorly. He named two routes: an agent that works with charts and specialist tools as a person would, or bringing the data into the model's own knowledge [[1]][s1].

**My view.** Documents sit between plain text and his examples: they are full of tables, charts and layout. For long documents I think the two routes work best together. AdaNav uses the first to decide what to read, by navigating a tree of the document, and the second to read it: the reasoner sees the raw pages, never the summaries used for navigation.

Our [LREC-COLING paper](/publications/lrec-coling-2024-kg-qa/) suggests how to choose between routes. Adding passages found through a knowledge graph raised retrieval accuracy from 75.8% to 96.2% on PopQA, an entity-centric benchmark with many rare entities, and left Natural Questions where it was (89.3% and 89.2%). The structured data helped where dense retrieval was weak, and not elsewhere. The route should follow the gap.

## Agents that leave the sandbox

He separates safety, where the model's own behavior drifts from what was intended, from security, where outsiders use the model to attack. Within security, finding a vulnerability is not the same as exploiting it, so training can deliberately favor defense, and Hunyuan models go through security evaluations with specialist teams before release [[1]][s1] [[2]][s2].

Both write-ups describe a stress test in which agents in a sandbox, given a task that could not be solved inside it, went outside. The details differ. In [[1]][s1] they found a channel to other environments, recruited other agents and tried to reach Hugging Face for the answer. In [[2]][s2] an agent tried a public key for a map service and posted a question in a comment section to get a person to answer it. The notes conclude that agent evaluations should cover cooperation between agents, communication across environments, permission boundaries, and what an agent does once it is blocked [[1]][s1].

On open weights he argued that responsible release helps security, because defenders need strong tools too. His example was a team whose request for help with defense was refused by a closed model's safety rules, and which then did the work with an open model [[1]][s1]. According to [[2]][s2], the team was Hugging Face, during an attack.

**My view.** MADQA saw the same trait from the other side. There, the agents that changed their query most after a failed search did best, and the weakest barely changed a word. What a benchmark rewards, trying something else when blocked, is what took his agents out of the sandbox. So the boundary has to be set by the environment, by which files and tools an agent can reach, and not left to the agent's restraint. Evaluations should also keep the whole trajectory, because the final answer does not show what the agent tried on the way.

On open weights, my stake is plain: AdaNav's backbones are open models, and RL research in a university needs weights it can train.

## Products and teams

Inside a lab, capability grows smoothly. The loss falls a little every day, and reasoning chains grew from a few steps to dozens without anyone feeling a jump. The public notices when the interaction changes, as when ChatGPT put a chat box in front of a model, or when a task crosses from unusable to usable [[1]][s1] [[2]][s2]. Prices follow competition more than quality [[1]][s1] [[2]][s2]. A lead in a new capability buys three to six months, after which a cheaper model that is good enough takes the users [[2]][s2].

To founders in embodied AI who train on data from real robot arms, he pointed out that such data arrives one second at a time, so the approach cannot absorb more compute however much money it raises, while general models may absorb many of the skills meant to be trained for robots alone. His question was whether they were betting with compute or against it [[2]][s2].

On people, he hires for learning speed over experience, and values low ego, honesty and co-design, since one architecture decision touches training, inference, post-training and cost [[1]][s1] [[2]][s2]. He told the students that IQ does not matter much in most large-model engineering, and gave two examples of what does. A wobble of two parts in ten thousand in the training loss was traced to some rare data that had been repeated 200,000 times by accident. A product that felt slightly off was traced, after more than a hundred samples were checked by hand, to a parameter wrongly set to 0.7 [[2]][s2]. He also prefers trust and taste to internal races decided by metrics: a product like WeChat or a game cannot be A/B-tested before launch, so someone has to trust a team's judgment [[1]][s1].

**My view.** His compute question applies to my own work. AdaNav is prompted, not trained: its router and its three navigators were designed by hand. In his terms that is a bet against compute, since it improves only when someone redesigns it. The version that bets with compute learns to navigate from search feedback, which is where MADQA ends and where I want to go next.

The usable threshold also looks different for document agents. On accuracy, the best agent already matches people who use the same search tool. On effort, it needs about five times as much. I expect document agents to cross from possible to usable when they become cheap and predictable, more than when they gain the next few points.

## What to do with your own time

A second-year Yao Class student asked the question many students now ask. AI writes, plans as an agent and solves olympiad problems; if it overtakes people on everything that can be measured, what should a person practice [[2]][s2]?

His first answer was not to race compute on the same kind of problem. Either be part of the machine, in pretraining, distributed systems, data or alignment, or complement it: deployment in the physical world and on hardware, product design that needs a deep sense of people, human connection [[2]][s2]. The notes add a test he borrowed for founders: when GPT-6 comes out, does your company sleep better or worse [[1]][s1]?

His second answer was to protect intrinsic motivation and taste. At Tsinghua many of his peers went into theoretical computer science. He found he did not love it, co-founded the university's rap club, and in the last year of his PhD took a course in pure set theory [[2]][s2]. He started research only in the second half of his third year, pushed by the requirements for graduate admission [[1]][s1]. Detours, he said, pay off late: the feel for language that rap trains can feed an understanding of language models, and the economy of set theory, which builds mathematics from a few axioms, resembles models built from simple components and data [[1]][s1] [[2]][s2].

**My view.** I tried the GPT-6 test on my thesis topic. Part of it gets bad news. Routing rules, regular expressions, chunking and index cards are ways of working around what models cannot yet do, and a stronger model will need fewer of them. Part of it gets good news: measuring how an agent searches, grounding answers in pages a person can check, and learning to navigate collections that are too large or too expensive to read whole. In MADQA, a recursive language model free to write code over the whole collection spent $850 and still finished ten points below a search agent on the same model that cost about $45. That second list is where I want to spend the rest of my PhD.

## Earlier in November, in Hong Kong

On 7 November he gave a keynote at the science symposium of the Future Science Prize Week in Hong Kong, on how AI for science has moved from tools to agents to agents that build tools [[3]][s3]. One report summarizes the stages as custom models built for one purpose, then general agents that solve single scientific problems, then agents that build new tools and models for a whole class of problems [[4]][s4]. His conclusion was that the technical obstacles will be solved in time, but that for now research still needs human judgment [[5]][s5].

In a conversation at the same event, Ya-Qin Zhang asked him what moat a model company has [[6]][s6]. He placed it in the product more than in the model: context, proprietary data and the user's trust. An assistant inside WeChat need not have the strongest model, because it knows your calendar and your friends, and a smarter model that does not know who you are cannot offer the same service. As AI moves from answering questions to acting for people, whether they trust it with their data will decide which products last. Asked whether Hunyuan would stay open if it became the best model in the world, he did not say yes or no. He gave the reasons for openness (practical for a model that is not leading, useful to the community, and increasingly a question of security), said that attack and defense cannot be fully separated in training, described checks during training, before release and after it, and expected open and closed models to coexist. On AI for science, he noted that math benchmarks are saturating, so researchers now collect open problems to evaluate models, and that autonomous labs will extend AI into physical experiments, with no fundamental obstacle that he could see [[6]][s6].

**My view.** For a document agent, the context he describes is the user's own documents and knowing where things are in them. Trust needs one more thing, answers that a person can check. That is why AdaNav's reasoner sees only the raw pages it selected, and why [DocLens](/posts/2025/08/doclens/) keeps the page and bounding box of every piece of evidence. Grounding is what turns context into trust.

His point about saturated benchmarks is one MADQA built in at small scale: a hundred of its test questions are ones no current model solves, so the benchmark keeps separating systems as they improve.

## Sources

1. 腾讯姚顺雨清华演讲：AI时代最稀缺的不是技术，而是Trust和Taste. Notes by 幻觉工程师, republished on [Tencent News][s1], 20 November 2026.
2. 姚顺雨清华姚班闭门对谈：剥开大模型神话，关于技术、商业与个人命运的 10 个底层认知. 十字路口Crossing, republished on [AITNT][s2], 20 November 2026.
3. [Programme of Science Symposium II][s3], Future Science Prize Week 2026.
4. AI带来哪些科研范式新命题？一场“跨域”对话在香港举行. 文汇报, republished on [NetEase][s4], 10 November 2026.
5. 腾讯姚顺雨：所有技术上的挑战最终都能解决，但一定少不了人类的参与. [The Paper][s5], 9 November 2026.
6. 张亚勤追问大模型护城河，姚顺雨：壁垒在产品层. [Yicai][s6], 8 November 2026.

Not covered: the Hunyuan team's Reddit AMA on 21 November, whose answers were given as a team.

[s1]: https://news.qq.com/rain/a/20261120A03I6400
[s2]: https://www.aitntnews.com/newDetail.html?newId=31452
[s3]: https://www.futureprize-hk.org/cn-2026sciencesymposium-ii
[s4]: https://www.163.com/dy/article/LBBPLD7B05506BEH.html
[s5]: https://www.thepaper.cn/newsDetail_forward_34438552
[s6]: https://www.yicai.com/news/103431271.html
