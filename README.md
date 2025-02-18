# EAGLE
Efficient Approach for Guided Local Examination in Digital Pathology

[Preprint](https://arxiv.org/abs/xxx) | [Cite](#citation)

## Abstract
>Artificial intelligence has transformed digital pathology by enabling biomarker prediction from high-resolution whole slide images (WSIs). However, current methods are computationally intensive, processing thousands of redundant tiles per WSI and requiring complex aggregator models. We introduce EAGLE (Efficient Approach for Guided Local Examination), a framework that emulates pathologists by selectively analysing informative regions. EAGLE incorporates two state-of-the-art foundation models: CHIEF for efficient tile selection and Virchow2 for extracting high-quality features from these tiles. Benchmarking was conducted against leading slide- and tile-level foundation models across 31 tasks from four cancer types, spanning morphology, biomarker prediction and prognosis. Achieving an average AUROC of 0.742, EAGLE matches the best slide encoder TITAN at 0.740 and surpasses the best tile-level model Virchow2 at 0.723. In biomarker prediction tasks, it reaches an AUROC of 0.772 compared to TITAN's 0.763 and Virchow2's 0.744. EAGLE processes a slide in 2.27 seconds, requiring only 1.1% of time compared to TITAN’s tile encoder. This computational advantage facilitates real-time workflows, and enables pathologists to easily validate all tiles which are used by the model during analysis. By reliably identifying meaningful regions and minimizing artifacts, EAGLE provides robust and interpretable outputs, supporting rapid slide searches, integration into multi-omic pipelines and emerging clinical foundation models.

<p align="center">
    <img src="assets/fig1v2hd.png" alt="failed loading the image" width="1100"/>
</p>

## Acknowledgements

We thank the authors and developers for their contribution as below.

- [STAMP](https://github.com/KatherLab/STAMP)
- [CHIEF](https://github.com/hms-dbmi/CHIEF)
- [Virchow2](https://huggingface.co/paige-ai/Virchow2)

## Citation

If you find our work useful in your research or if you use parts of this code please consider citing our [preprint](https://arxiv.org/abs/xxx):

Neidlinger, P. et al. A deep learning framework for efficient pathology image analysis, _Arxiv_, 2025

```bibtex
@misc{neidlinger2025eagle,
      title={A deep learning framework for efficient pathology image analysis}, 
      author={Peter Neidlinger and Tim Lenz and Sebastian Foersch and Chiara M. L. Loeffler and Jan Clusmann and Marco Gustav and Lawrence A. Shaktah and Rupert Langer and Bastian Dislich and Lisa A. Boardman and Amy J. French and Ellen L. Goode and Andrea Gsur and Stefanie Brezina and Marc J. Gunter and Robert Steinfelder and Hans-Michael Behrens and Christoph Röcken and Tabitha Harrison and Ulrike Peters and Amanda I. Phipps and Giuseppe Curigliano and Nicola Fusco and Antonio Marra and Michael Hoffmeister and Hermann Brenner and Jakob Nikolas Kather},
      year={2025},
      eprint={xxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/xxx}, 
}
```
