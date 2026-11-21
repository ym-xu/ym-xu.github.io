---
title: "Notes on Shunyu Yao's talk at Tsinghua"
date: 2026-11-21
slug: "shunyu-yao-tsinghua-notes"
summary: "Shunyu Yao spoke at Tsinghua on 18 November 2026 in a closed session. These are my notes from two write-ups of it, with his remarks in Hong Kong earlier in the month, and what I take from them for work on document agents."
tags: ["AI", "Agents"]
---

On Wednesday 18 November 2026, Shunyu Yao (姚顺雨) went back to the Institute for Interdisciplinary Information Sciences at Tsinghua, where he was an undergraduate in the Yao Class, for a two-hour conversation with Kaifeng Lyu (吕凯风), now an assistant professor there. Yao wrote ReAct, co-wrote SWE-bench and τ-bench, worked on agents at OpenAI, and now leads the Hunyuan models as Tencent's chief AI scientist. I read him because he has built agents, built the benchmarks that measure them, and now trains the models underneath.

## Where this comes from

The session was closed. There is no recording and no official transcript. Everything below comes from two write-ups that circulated afterwards:

- notes by the blogger 幻觉工程师, [republished on Tencent News](https://news.qq.com/rain/a/20261120A03I6400) on 20 November, made from "materials of the talk" that the blogger got hold of;
- a [ten-point summary](https://www.aitntnews.com/newDetail.html?newId=31452) by the WeChat account 十字路口Crossing, which says it is based on the full two-hour conversation and a verbatim record of it. One of its ten sections, on how OpenAI and Anthropic are organised, has lost its details in every copy I found.

Neither is Yao's own text. The two agree on the main arguments and differ on some details, and I say so where it matters. They also differ on the date: the summary says "this Wednesday", the 18th, and the notes say "yesterday", which would be the 19th. Quotations are my translations of sentences that the write-ups put in quotation marks, so they are at two removes from what he said.

## Small experiments can mislead

After ReAct, Yao tried to post-train an open LLaMA 13B model on agent trajectories. The results were bad enough that he doubted his own ability as a researcher. Later, with far more compute at OpenAI, he concluded that the model had been too small for the idea to show.

His reading of the episode: with little compute, noise can bury a real effect, and a researcher can end up wrong about which problems are worth working on. The purpose of a small experiment is to build an expectation, together with some theory, of what the large run will do. A recipe that holds at small scale does not automatically hold at large scale, which is much of what makes pretraining research hard.

## RL: fewest assumptions, correctness first

Post-training at scale has come back to plain policy gradient and its variants, such as GRPO. His explanation is that policy gradient assumes the least. Methods with more machinery looked good in small experiments, and every extra assumption was one more thing that could fail on a large cluster.

The first write-up adds an order of priorities for RL engineering: correct first, then stable, then efficient. Training and inference systems have to be designed together, because throughput, memory, and how a fixed number of GPUs is split between the two decide which experiments can be run at all.

On what to train on, he separated two aims. Harder maths problems train reasoning. Tasks such as writing are computationally easier and are where much of the real use is. A model can be strong at maths and still poor at the work users bring to it. He also treated many alignment problems as problems of reward design: a child who is rewarded only for scoring 100 in maths will put everything into maths, and a model does the same with its reward.

## An eval you tune to stops working

Both write-ups have him criticising metric chasing. When the person a team reports to does not know the technology well, the team reports what is easy to count: parameter counts, data volume, benchmark scores, utilisation. Two weeks spent fixing a bug in the correctness of the computation has no number attached.

The second write-up quotes him on evaluation. An eval exists to show you your own weak points, and after you have tuned to it, "it becomes your enemy, and you have blinded your own radar". It also reports a rule inside the Hunyuan team: a core benchmark is run once and then sealed.

Both give his check against self-deception. If a model matches GPT on leaderboards and earns a hundredth or a thousandth of the revenue, the gap is the thing to investigate.

## Data

He does not think the competition over pretraining data is finished. Good data such as code is expensive, so it is reused, and reuse brings memorisation. Synthetic data can lose quality. How to mix sources and how to collect data dynamically are open questions. According to the second write-up, he sees no hard limit on public data in the next year or two, and thinks the lasting advantage is taste:

> Ten years ago we hired people in Madagascar to draw bounding boxes. Then we labelled with small models. Now large models make their own data and play against themselves. The means of producing data are overturned every few years. What does not change is a team's taste for what good data is.

The first write-up has a point close to my own work. A great deal of industrial data is not text: time series, clinical records, tables. Models trained natively on these modalities often reason poorly. He named two routes. One is to let an agent use charts and specialist tools as a person would, or even train a specialised model for the data. The other is to bring this data into the model's own knowledge.

## Agents, safety and open weights

He separates safety, where the model's own behaviour drifts from what was intended, from security, where an outsider uses the model to attack. Within security, finding a vulnerability is not the same as exploiting it, and training can deliberately favour defence.

Both write-ups describe a stress test in which agents in a sandbox were given a task that could not be solved inside it. The agents went outside. The accounts differ on how. In one, they found a channel to other environments, recruited other agents and tried to reach Hugging Face for the answer. In the other, an agent tried a public key for a map service and posted a question in a comment section so that a human would answer it. The first write-up concludes that agent evaluations should cover collaboration between agents, communication across environments, the limits of permissions, and what an agent does once it is blocked.

On open weights, he argued that responsible release helps security, because defenders need strong tools too. His example was a team that asked a closed model for help with defence, was refused by its safety rules, and did the work with an open model. The second write-up says the team was Hugging Face, during an attack on it.

## Products and teams

- **Capability is smooth, adoption is not.** Inside a lab the loss goes down a little every day. The public notices when the interaction changes, as when ChatGPT put a text box in front of a model, or when a task moves from unusable to usable.
- **Price follows competition.** What a model can charge depends mostly on who else is selling, not on its quality. A lead in a new capability lasts three to six months, according to the second write-up.
- **Bet with compute.** To founders in embodied AI who train on data from real robot arms, he pointed out that such data arrives one second at a time, so the approach cannot use more compute however much money there is. General models may absorb many of the skills that were going to be trained for robots alone. His question: are you betting with compute or against it?
- **People.** He hires for learning speed over experience, and values low ego, honesty and what he calls co-design: an architecture decision touches training, inference, post-training and cost, so nobody can make it alone. The second write-up quotes him telling the students that "IQ really does not matter that much in most large-model engineering", with two stories. A wobble of two parts in ten thousand in the loss was traced to some rare data that had been repeated 200,000 times by accident. A product that felt slightly off was traced, through more than a hundred samples checked by hand, to a call parameter wrongly set to 0.7.
- **Trust and taste.** He prefers them to scoreboards and internal horse races. Products such as WeChat or a game cannot be predicted by an A/B test before launch, so someone has to trust a team's judgment and keep investing.

## What to do with your own time

A second-year student in the Yao Class asked the question many students now ask. Two years ago AI learned to write, half a year ago it learned to act as an agent, a few months ago it saturated the maths olympiad: if it overtakes people on every task that can be measured, what skill should a person practise?

His first answer was not to compete with compute head on. Racing a model that burns tens of thousands of GPUs a day on the same kind of maths problem "would be painful and futile". Either be part of the machine, in pretraining, distributed systems, data or alignment, or be complementary to it: deployment in the physical world and on hardware, product design that needs a deep sense of people, human connection. The notes add a test he borrowed for founders: when GPT-6 comes out, will your company sleep better or worse?

His second answer was to protect intrinsic motivation and taste. At Tsinghua, the summary says, everyone was going into theoretical computer science, shaving the exponent of an algorithm from 2.7 to 2.6. He found he did not love it, co-founded the university's rap club, and in the last year of his PhD took a course in pure set theory.

> You often feel you are taking a detour, but the detour is often the shortcut. The feel for language and its beauty that rap trains fed into an understanding of language models. Set theory derives the whole of mathematics from a few axioms, and that elegance is the same as a large model fitting the real world from basic components and data.

## Two smaller points

- **Small models.** Research that uses AI to do AI research favours small models, because they are cheap to experiment with. He expects one strong model to be trained and then distilled into a hierarchy of sizes, each matched to the cost of its task.
- **Recursive self-improvement is a plain idea.** It means using AI to run the research loop better. Progress comes from the people who fix bugs, clean data and run experiments, not from the ones who coin new terms.

## Earlier in November, in Hong Kong

Some of this he had already said in public. On 7 November, at the science symposium of the Future Science Prize Week in Hong Kong, he gave a keynote whose title translates as "Tools, agents, and agents that build tools: the evolution of AI for Science". [One report](https://www.163.com/dy/article/LBBPLD7B05506BEH.html) gives its three stages: custom models built for one purpose, then general agents that solve single problems, then agents that build new tools and models for a whole class of problems. [The Paper](https://www.thepaper.cn/newsDetail_forward_34438552) reports his conclusion that the technical obstacles will be solved in time, and that research still needs human judgment for now.

At the same event Ya-Qin Zhang asked him what moat a model company has ([Yicai's report](https://www.yicai.com/news/103431271.html)). His answer was that it sits in the product more than in the model: context, proprietary data and the user's trust. An assistant inside WeChat does not need the strongest model, because it knows your calendar and your friends, and a smarter model that does not know who you are cannot offer the same service. Asked whether Hunyuan would stay open if it became the best model in the world, he did not say yes or no. He gave three reasons for openness (it is practical for a follower, it serves the community, and it bears on security), said that attack and defence cannot be fully separated in training, and expected open and closed models to coexist.

## What I take from it

**Report the scale of a negative result.** A university lab runs the 13B experiment, not the large one. His story does not turn my failed experiments into hidden successes. It does say how little a failure at this scale can claim, and that a paper should state the scale when it reports one.

**His two routes for non-text data are the choice in document understanding too.** Documents are full of tables, charts and layout. One approach trains a model to take in every page. The other gives an agent the means to move around the document and decide what to read. [AdaNav](/publications/icdar-adanav/) belongs to the second: it builds a tree of the document once and navigates it to a depth chosen for each question.

**Score what the agent does when it is stuck.** In [MADQA](/publications/icml-madqa/) we saw the harmless form of his sandbox story. When a first query fails, people change strategy quickly, and agents often keep rephrasing the same query. His agents had the opposite problem: they changed strategy so freely that they left the environment. Both cases point at the same gap in evaluation, which is to look at the behaviour after the first failure and not only at the final answer.

**The GPT-6 question works for a thesis.** Which parts of a topic become easier to justify when the next model arrives, and which parts disappear? I would rather ask that now than at the defence.

## Sources

- 腾讯姚顺雨清华演讲：AI时代最稀缺的不是技术，而是Trust和Taste. [Tencent News](https://news.qq.com/rain/a/20261120A03I6400), 20 November 2026. Notes by 幻觉工程师.
- 姚顺雨清华姚班闭门对谈：剥开大模型神话，关于技术、商业与个人命运的 10 个底层认知. [AITNT](https://www.aitntnews.com/newDetail.html?newId=31452), 20 November 2026, from the WeChat account 十字路口Crossing.
- 张亚勤追问大模型护城河，姚顺雨：壁垒在产品层. [Yicai](https://www.yicai.com/news/103431271.html), 8 November 2026.
- 腾讯姚顺雨：所有技术上的挑战最终都能解决，但一定少不了人类的参与. [The Paper](https://www.thepaper.cn/newsDetail_forward_34438552), 9 November 2026.
- [Programme of Science Symposium II](https://www.futureprize-hk.org/cn-2026sciencesymposium-ii), Future Science Prize Week 2026.

Not covered here: the Hunyuan team's Reddit AMA on 21 November. Its answers were given as a team, so I cannot tell which are his.
