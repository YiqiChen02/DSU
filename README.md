# Dynamic Snake Upsampling Operator and Boundary-Skeleton Weighted Loss for Curvilinear Structure Segmentation

This repository contains the implementation, dataset, and documentation for the paper:

“Dynamic Snake Upsampling Operator and Boundary-Skeleton Weighted Loss for Curvilinear Structure Segmentation”
Authors: Yiqi Chen, Ganghai Huang*, Sheng Zhang, Jianglin Dai


## 📜 Abstract

Accurate segmentation of curvilinear structures (e.g., fractures and vascular networks) is critical for reliable downstream analysis and modeling. However, in dense prediction tasks such as semantic segmentation and super-resolution, conventional upsampling operators struggle to reconstruct features that are both slender and highly curved.

This study introduces a **dynamic upsampling operator** and a **boundary-skeleton weighted loss** tailored for curvilinear structures.

The proposed upsampling operator uses an **adaptive sampling domain** that adjusts its sampling stride conditioned on feature maps and selects subpixel sampling points along a **serpentine path**. This design enables more accurate reconstruction of curvilinear structures at the **subpixel level**.

Meanwhile, we propose a **skeleton-to-boundary weighted loss** that balances weight allocation between the main structure and the boundary. The weights are determined by mask class ratios and distance maps, preserving segmentation overlap while improving **topological continuity** and **boundary alignment**, with **negligible computational cost**.

Experiments across different domain datasets and backbone networks show that this **plug-and-play** dynamic snake upsampling operator and boundary-skeleton weighted loss can boost both **pixel-wise accuracy** and **topological consistency** of segmentation results.

**Keywords:** Dynamic Snake Upsampling; Weighted Loss; Curvilinear Structure Segmentation; Deformable Kernel; Precomputed Map.

## ✨ Key Innovations

This project focuses on addressing the challenges of segmenting slender and curvilinear structures by introducing:

* **Dynamic Snake Upsampling Operator:**
    * Utilizes an **adaptive sampling domain** conditioned on feature maps.
    * Employs a **serpentine path** to select subpixel sampling points for accurate reconstruction.
* **Boundary-Skeleton Weighted Loss:**
    * Balances weight allocation between the core structure (skeleton) and the boundary using **class ratios and distance maps**.
    * Significantly improves **topological consistency** and **boundary alignment** with minimal overhead.
* **Plug-and-Play Design:** The components can be easily integrated into existing deep learning frameworks.

## 📥 Repository Contents

This repository provides:

* **Source Code:** For training and evaluating the proposed model.
* **datasets:** Processed datasets included DeepCrack and DRIVE.

## Citation

If you use this repository and our methodology in your research, please cite our paper: (to be added)
## License

This project is licensed under the **MIT License**. See `LICENSE` for more details.

## 📧 Contact

For any questions or issues, feel free to reach out to the corresponding author:
* **Yiqi Chen (First Author):** 244811086@csu.edu.cn
* **Ganghai Huang (Corresponding Author):** huangganghai@csu.edu.cn
