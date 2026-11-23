# Experiment No: 13
# Title: Conditional Probability and Joint Distributions
# Aim: Represent uncertainty using probability tables and verify probability
#      rules computationally for hospital patient symptom and diagnosis data.
# Libraries: NumPy, Pandas, Matplotlib

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. Generate synthetic patient dataset ─────────────────────────
# Symptoms: Fever (0/1), Cough (0/1)
# Diagnoses: Flu (0/1), Cold (0/1)
n = 500

fever = np.random.binomial(1, 0.4, n)
cough = np.where(fever == 1,
                 np.random.binomial(1, 0.75, n),   # more likely if fever
                 np.random.binomial(1, 0.3,  n))

flu  = np.where((fever == 1) & (cough == 1),
                np.random.binomial(1, 0.70, n),
                np.random.binomial(1, 0.10, n))

cold = np.where((cough == 1) & (fever == 0),
                np.random.binomial(1, 0.65, n),
                np.random.binomial(1, 0.08, n))

df = pd.DataFrame({'Fever': fever, 'Cough': cough,
                   'Flu': flu, 'Cold': cold})
print("Dataset (first 10 rows):")
print(df.head(10))
print(f"\nTotal patients: {len(df)}")

# ── 2. Joint probability table (Fever × Cough) ────────────────────
joint_table = pd.crosstab(df['Fever'], df['Cough'],
                           rownames=['Fever'], colnames=['Cough'],
                           normalize=True)
print("\nJoint Probability Table P(Fever, Cough):")
print(joint_table.round(4))

# ── 3. Marginal probabilities ─────────────────────────────────────
p_fever1 = df['Fever'].mean()
p_cough1 = df['Cough'].mean()
p_flu1   = df['Flu'].mean()
p_cold1  = df['Cold'].mean()

print(f"\nMarginal Probabilities:")
print(f"  P(Fever=1) = {p_fever1:.4f}")
print(f"  P(Cough=1) = {p_cough1:.4f}")
print(f"  P(Flu=1)   = {p_flu1:.4f}")
print(f"  P(Cold=1)  = {p_cold1:.4f}")

# ── 4. Conditional probabilities ──────────────────────────────────
def conditional_prob(df, target_col, cond_col, target_val=1, cond_val=1):
    cond_mask = df[cond_col] == cond_val
    return df.loc[cond_mask, target_col].mean()

p_flu_given_fever = conditional_prob(df, 'Flu',  'Fever')
p_flu_given_cough = conditional_prob(df, 'Flu',  'Cough')
p_cold_given_cough = conditional_prob(df, 'Cold', 'Cough')
p_flu_given_both   = df.loc[(df['Fever']==1) & (df['Cough']==1), 'Flu'].mean()

print(f"\nConditional Probabilities:")
print(f"  P(Flu  | Fever=1)            = {p_flu_given_fever:.4f}")
print(f"  P(Flu  | Cough=1)            = {p_flu_given_cough:.4f}")
print(f"  P(Cold | Cough=1)            = {p_cold_given_cough:.4f}")
print(f"  P(Flu  | Fever=1, Cough=1)   = {p_flu_given_both:.4f}")

# ── 5. Verify probability rules ───────────────────────────────────
print("\n── Rule Verification ──")

# Multiplication rule: P(A ∩ B) = P(A|B) * P(B)
p_fever_and_flu = (df['Fever'] * df['Flu']).mean()
p_flu_given_fever_check = p_fever_and_flu / p_fever1
print(f"Multiplication rule: P(Fever∩Flu) = {p_fever_and_flu:.4f}")
print(f"  P(Flu|Fever)*P(Fever) = {p_flu_given_fever:.4f} × {p_fever1:.4f} "
      f"= {p_flu_given_fever*p_fever1:.4f}  ✓" if
      abs(p_flu_given_fever*p_fever1 - p_fever_and_flu) < 0.01
      else f"  Mismatch: {p_flu_given_fever*p_fever1:.4f}")

# Total probability: P(Flu) = P(Flu|Fever=1)*P(Fever=1) + P(Flu|Fever=0)*P(Fever=0)
p_flu_given_no_fever = conditional_prob(df, 'Flu', 'Fever', cond_val=0)
p_flu_total = (p_flu_given_fever * p_fever1 +
               p_flu_given_no_fever * (1 - p_fever1))
print(f"\nTotal probability rule:")
print(f"  P(Flu) direct          = {p_flu1:.4f}")
print(f"  P(Flu) via total rule  = {p_flu_total:.4f}  "
      f"{'✓' if abs(p_flu_total - p_flu1) < 0.02 else '≈'}")

# ── 6. Visualise conditional probabilities ─────────────────────────
conditions = ['Fever=1', 'Fever=0', 'Cough=1', 'Cough=0',
              'Fever+Cough=1', 'Neither']
flu_probs  = [
    conditional_prob(df, 'Flu', 'Fever', cond_val=1),
    conditional_prob(df, 'Flu', 'Fever', cond_val=0),
    conditional_prob(df, 'Flu', 'Cough', cond_val=1),
    conditional_prob(df, 'Flu', 'Cough', cond_val=0),
    df.loc[(df['Fever']==1) & (df['Cough']==1), 'Flu'].mean(),
    df.loc[(df['Fever']==0) & (df['Cough']==0), 'Flu'].mean()
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(conditions, flu_probs, color='steelblue', edgecolor='k')
axes[0].axhline(p_flu1, color='red', linestyle='--', label=f'P(Flu) = {p_flu1:.3f}')
axes[0].set_ylabel('P(Flu | Evidence)')
axes[0].set_title('Conditional Probability of Flu Given Symptoms')
axes[0].set_xticklabels(conditions, rotation=15, ha='right')
axes[0].legend()

# Heatmap: P(Flu | Fever, Cough)
grid = np.array([[df.loc[(df['Fever']==f) & (df['Cough']==c), 'Flu'].mean()
                  for c in [0,1]] for f in [0,1]])
im = axes[1].imshow(grid, cmap='YlOrRd', vmin=0, vmax=1)
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(['Cough=0','Cough=1'])
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(['Fever=0','Fever=1'])
axes[1].set_title('P(Flu | Fever, Cough) Heatmap')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center',
                     fontsize=14, color='black')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('lab13_conditional.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Joint probability table, marginal and conditional probabilities.
# Verification that multiplication rule and total probability rule hold.
# Bar chart and heatmap showing how symptoms change Flu probability.

# RESULT
# Conditional probability tables quantify the dependence between symptoms
# and diagnoses. The multiplication and total probability rules are verified
# numerically. Combined evidence (Fever AND Cough) dramatically increases
# the probability of Flu diagnosis compared to individual symptoms.
