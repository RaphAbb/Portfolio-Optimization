## MS&E 311 / CME 307: Optimization

# Most Diversified Portfolio

#### Raphael Abbou, Simon Hagege

##### Abstract

Finding the optimal investment portfolio has been a research topic for decades. This paper seeks to provide a detailed resolution of the Maximum Diversity Portfolio (MDP) problem introduced by T. Froidure, Y. Choueifaty, J. Reynier, which translates into a portfolio diversification problem into a quadratic convex optimization one. We develop two approaches of the problem, depending on whether or not short-selling is allowed. The short-allowed strategy can be solved analytically while the long-only approach is obtained using an Alternating Direction Method of Multipliers. We study and report each portfolio performance on ten ETF across five asset classes, backtesting these over ten years against the MSCI World Index.

##### Results

| **Statistic** | **MSCI** | **Inverse Vol** | **MDP Analytical** | **ADMM MDP** |
| :------------ | :------: | :-------------: | :----------------: | :----------: |
| Sharpe LY     |  0.724   |      2.736      |       2.070        |              |
| Sharpe L2Y    |  0.556   |      1.939      |       2.264        |              |
| Sharpe L5Y    |  0.189   |      0.583      |       0.592        |              |

##### Course

This project was part of the course MS&E 311 / CME 307: Optimization taught at Stanford University.
https://web.stanford.edu/class/msande311/

