const COMMON_PILE_TEXT_HEAD: &str = "# Probing cosmic strings

Zehua Tian, Email: tianzh@ustc.edu.cn CAS Key Laboratory of
Microscale Magnetic Resonance, Hefei 230026, China

###### Abstract

Quantum metrology studies the ultimate precision limit
of physical quantities.

CAS Key Laboratory, Hefei 230026, China (tianzh@ustc.edu.cn)

##  1 Introduction

Cosmic strings are line-like
defects. ";

const COMMON_PILE_TEXT_TAIL: &str = "

######  Lemma 1

The deficit angle is small. Fig. 3\\. shows $\\Delta_{12}$.

  * first point
  * second point

## References

  * (1) M. S. Turner, Science 262, 861 (1993).
";

pub fn common_pile_text() -> String {
    format!("{COMMON_PILE_TEXT_HEAD}{}{COMMON_PILE_TEXT_TAIL}", "They bend light. ".repeat(120))
}
