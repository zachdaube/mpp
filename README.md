# Molecular Property Prediction with Iterative Model Design

## Overview
The primary job of ML researchers at Bindwell is to repeat: design, implement, evaluate, then analyze models. That’s what we’ll test here. You’ll model lipophilicity (a key property in chemical discovery) using a standard benchmark and build tooling for rapid iteration. We’re evaluating your ability to own high-impact ML research: architecture design, coding, and analysis skills you’ll use daily on our team.

## Requirements
- Use the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/start/) library to load the [Lipophilicity_AstraZeneca benchmark](https://tdcommons.ai/single_pred_tasks/adme/#lipophilicity-astrazeneca). Stick to the provided splits (scaffold-based) and evaluate using Mean Absolute Error (MAE). If you're not on linux (meaning you can't install pytdc) or don't want to use the package for whatever reason, you can use the csv files in the data folder, it's the same data.
- Develop your own model architectures for this regression task. Start with a baseline and iterate on at least 3 variants (e.g., classic ML, graph-based, sequence-based, hybrid approaches, etc.).
- Build a platform for rapid iteration: Create a modular system to experiment with architectures, hyperparameters, and seeds. Ensure reproducibility and efficient logging of results.
- Analyze results: Generate at least 3-5 plots exploring relationships between architectural choices (e.g., parameter count, architecture, mechanisms, etc.) and val model accuracy (MAE). Include a brief discussion of insights and potential model weaknesses and trends.
- Deliverables:
  - A zip file with all code, plots, and a report (pdf, markdown, or jupyter notebook (ipynb)) summarizing approach, results, and findings.
  - README with setup instructions, dependencies, and reproduction steps.
- Time estimate: 8-10 hours. You will be compensated with money upon completion.

## Guidelines and Tips:
- Use AI to help you.
- Keep track of how much time you spend on this.
- Use pytorch, or even better, pytorch lightning.
- If you're running into compute issues, try vast.ai or colab.
- Work in a virtual environment.
- You can make use of pretrained huggingface models if you want
- Wandb is optional but recommended.
- If you use any open-source snippets or designs from papers, you must cite them clearly (but keep in mind we want to assess **your** ideas and skills).
- Check the [TDC leaderboard](https://tdcommons.ai/benchmark/admet_group/05lipo/) to get a sense of the state of the art.
- Code should be self contained in the submitted zip file. Make sure setup is easy.

## Evaluation Criteria
- **Iteration Platform**: Effectiveness in supporting quick experiments (modularity, logging, reproducibility).
- **ML Iteration Strategy**: Diversity and relevance of architectures tested, logical design strategy.
- **Analysis**: Quality of plots and insights connecting design to performance.
- **Code Quality**: Cleanliness, documentation, and engineering best practices.
- **Model Accuracy**: Best MAE achieved on test set.

Submit zip file via email to tyler@bindwell.ai.

## License Grant
By commencing work on this assignment, you acknowledge that you have understood and agreed to the following terms regarding the deliverables you submit:

- **Ownership Retention**: You retain all right, title, and interest in and to any code, models, architectures, analyses, reports, plots, and other materials contained in the final submitted zip file (collectively, "Submitted Materials"), including all intellectual property rights therein (e.g., copyrights, patents, trade secrets).
- **License Grant to Bindwell**: You hereby grant Bindwell a worldwide, perpetual, irrevocable, royalty-free, non-exclusive license to use, reproduce, modify, adapt, distribute, perform, display, create derivative works from, and otherwise exploit the Submitted Materials for any purpose, including commercial, internal, or external use, without any further obligation or compensation to you. This license includes the right to sublicense these rights to third parties (e.g., affiliates or partners) at Bindwell's discretion.
- **No Restrictions on Your Rights**: Except as limited by the Non-Disclosure terms (e.g., you may not publicly disclose assignment details or share the Submitted Materials in a way that reveals confidential information or references Bindwell), this grant does not restrict your ability to use, license, or otherwise exploit the Submitted Materials for your own purposes, including public sharing or licensing, provided you remove or obscure any references to Bindwell or the assignment's confidential details and present it in a generic manner.
- **Representations**: You represent that the Submitted Materials are original to you (except for properly cited open-source materials) and do not infringe third-party rights. You agree to cooperate with Bindwell (at Bindwell's expense) to enforce or defend these rights if needed.
- **Waiver of Certain Rights**: To the extent permitted by law, you waive any moral rights (e.g., attribution or integrity) solely with respect to Bindwell's use of the Submitted Materials under this license.

If you do not agree to these terms, do not proceed with the assignment.