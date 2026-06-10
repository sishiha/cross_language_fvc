# Tools for Speech processing and forensic voice comparison 
This repository contains tools for speech signal processing, including implementations of the band-limited cepstral coefficient (BLCC) method and experimental tools for forensic voice comparison (FVC). These tools support the cross-language FVC project.

Cross-language FVC refers to forensic-oriented speaker verification involving speech samples produced in different languages, such as French vs. Thai. *To improve cross-language FVC performance, the present project uses the BLCC method to investigate how linguistic-phonetic information and speaker-specific information are encoded across the spectral range, and how their distributions vary across languages.* The findings are expected to provide insights into how language-dependent phonetic information can be suppressed while maximising speaker-specific information, thereby improving the robustness of cross-language FVC.

- For the BLCC tools, refer to `BLCC_EXTRACTION.md`
- For the FVC experiments, refer to `FVC_EXPERIMENTS_WITH_CCS.md`
