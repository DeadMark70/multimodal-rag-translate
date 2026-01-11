"""Test fix_malformed_latex function strict edge cases."""
from pdfserviceMD.markdown_cleaner import fix_malformed_latex

print("--- Testing specific sub-environments ---")
# 1. Aligned - MUST wrap
t_aligned = r'\begin{aligned}a &= b\end{aligned}'
r_aligned = fix_malformed_latex(t_aligned)
print(f"Orphan aligned: {'PASS' if '$$' in r_aligned else 'FAIL'} -> {r_aligned}")

# 2. Equation - MUST NOT wrap
t_equation = r'\begin{equation}E=mc^2\end{equation}'
r_equation = fix_malformed_latex(t_equation)
print(f"Orphan equation: {'PASS' if '$$' not in r_equation else 'FAIL'} -> {r_equation}")

# 3. Align - MUST NOT wrap (top level)
t_align = r'\begin{align}a &= b\end{align}'
r_align = fix_malformed_latex(t_align)
print(f"Orphan align: {'PASS' if '$$' not in r_align else 'FAIL'} -> {r_align}")

print("\n--- Testing whitespace handling ---")
# 4. Aligned with preceding space and $$ - MUST NOT wrap
t_spaced = r'$$   \begin{aligned}a &= b\end{aligned}   $$'
r_spaced = fix_malformed_latex(t_spaced)
# It should remain exactly as is (no extra $$), but whitespace might be normalized
print(f"Spaced aligned: {'PASS' if '$$' in r_spaced and r_spaced.count('$$') == 2 else 'FAIL'}")
if r_spaced != t_spaced:
    print(f"   Expected: {t_spaced}")
    print(f"   Got:      {r_spaced}")

# 5. Multiline context
t_multiline = r'''
$$
\begin{aligned}
x &= y
\end{aligned}
$$
'''
r_multiline = fix_malformed_latex(t_multiline)
# Check for existing $$ logic preservation
print(f"Multiline aligned: {'PASS' if '$$' in r_multiline and r_multiline.count('$$') == 2 else 'FAIL'}")

# 6. Orphan with text
t_orphan_text = r'Here is text \begin{aligned}x\end{aligned} end.'
r_orphan_text = fix_malformed_latex(t_orphan_text)
print(f"Text + Orphan: {'PASS' if '$$\\begin' in r_orphan_text else 'FAIL'} -> {r_orphan_text}")

print("\n--- Testing nested environment handling (Bug Fix) ---")
# 7. Aligned inside Equation - MUST NOT wrap
t_nested = r'''
\begin{equation}
    \begin{aligned}
        x &= y
    \end{aligned}
\end{equation}
'''
r_nested = fix_malformed_latex(t_nested)
if "$$" in r_nested:
    print(f"Nested Equation+Aligned: FAIL (Wrapped in $$)")
    print(r_nested)
else:
    print(f"Nested Equation+Aligned: PASS (Not wrapped)")

# 8. Aligned inside $$ - MUST NOT wrap (Double check)
t_nested_dollar = r'''
$$
\begin{aligned}
    x &= y
\end{aligned}
$$
'''
r_nested_dollar = fix_malformed_latex(t_nested_dollar)
if r_nested_dollar.count("$$") > 2:
    print(f"Nested $$+Aligned: FAIL (Double wrapped)")
else:
    print(f"Nested $$+Aligned: PASS (Not double wrapped)")
