# EAGLE
Efficient Approach for Guided Local Examination in Digital Pathology

[Preprint](https://arxiv.org/abs/xxx) | [Cite](#citation)

## Abstract
>Artificial intelligence has transformed digital pathology by enabling biomarker prediction from high-resolution whole slide images (WSIs). However, current methods are computationally intensive, processing thousands of redundant tiles per WSI and requiring complex aggregator models. We introduce EAGLE (Efficient Approach for Guided Local Examination), a framework that emulates pathologists by selectively analysing informative regions. EAGLE leverages CHIEF for efficient tile selection and Virchow2 for high-quality feature extraction from these tiles. In evaluations involving 9,528 WSIs from 6,818 patients across 13 cohorts and four tissue sites, EAGLE achieves an average AUROC of 0.742, outperforming the best slide encoder, TITAN (0.740), and the best tile-level foundation model, Virchow2 (0.723). In biomarker prediction tasks, EAGLE reaches an AUROC of 0.772 compared to TITAN's 0.763 and Virchow2's 0.744. EAGLE processes a slide in 2.27 seconds, requiring only 1.1% of time compared to TITAN’s tile encoder. This computational advantage facilitates real-time workflows, and enables pathologists to easily validate all tiles which are used by the model during analysis. By reliably identifying meaningful regions and minimizing artifacts, EAGLE provides robust and interpretable outputs, supporting rapid slide searches, integration into multi-omic pipelines and emerging clinical foundation models.

<p align="center">
    <img src="assets/fig1.png" alt="failed loading the image" width="1100"/>
</p>

## Citation

If you find our work useful in your research or if you use parts of this code please consider citing our [preprint](https://arxiv.org/abs/xxx):

Neidlinger, P. et al. Realising CHIEF's Promise: EAGLE Enhances WSI Classification Through Efficient Tile Selection, _Arxiv_, 2025

```bibtex
@misc{neidlinger2025eagle,
      title={Realising CHIEF's Promise: EAGLE Enhances WSI Classification Through Efficient Tile Selection}, 
      author={Peter Neidlinger and Tim Lenz and Sebastian Foersch and Michael Hoffmeister and Hermann Brenner and Chiara Maria Lavinia Loeffler and Jan Clusmann and Rupert Langer and Bastian Dislich and Hans Michael Behrens and Christoph Röcken and Antonio Marra and Jakob Nikolas Kather},
      year={2025},
      eprint={xxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/xxx}, 
}
```
