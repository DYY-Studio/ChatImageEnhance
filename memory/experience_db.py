import chromadb
import numpy as np
import cv2
import json
import uuid
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer 

class ExperienceDB:
    def __init__(self, db_path="./chroma_db"):
        """初始化 ChromaDB 客户端和集合"""
        self.client = chromadb.PersistentClient(path=db_path)
        # 使用余弦相似度进行距离计算 (hnsw:space = cosine)
        self.collection = self.client.get_or_create_collection(
            name="image_enhance_experience",
            metadata={"hnsw:space": "cosine"} 
        )
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2') # 生产环境文本模型

    def _get_image_embedding(self, img: np.ndarray) -> list[float]:
        """
        提取多维度的传统图像特征，构建更适合图像增强领域的 Embedding 向量。

        包含：色彩直方图、光照统计量、锐度评估、边缘与噪声代理特征。
        """
        # 将图像调整为固定大小以统一直方图和像素统计的尺度
        img_resized = cv2.resize(img, (256, 256))
        features = []

        # ==========================================
        # 1. 色彩分布特征 (HSV 直方图) 
        # ==========================================
        # 捕捉整体偏色、过饱和或色彩匮乏 (尺寸: 16x16 = 256 维)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # 为了平衡不同特征的权重，可以对高维度的直方图进行一定缩放 (经验值)
        features.extend((hist.flatten() * 0.5).tolist())

        # ==========================================
        # 2. 光照与对比度特征 (LAB 亮度通道统计)
        # ==========================================
        # 捕捉图像是过曝、欠曝、还是对比度极低 (尺寸: 4 维)
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        l_mean = np.mean(l_channel) / 255.0       # 平均亮度
        l_std = np.std(l_channel) / 255.0         # 整体对比度
        l_p10 = np.percentile(l_channel, 10) / 255.0  # 暗部极值（用于判断死黑）
        l_p90 = np.percentile(l_channel, 90) / 255.0  # 亮部极值（用于判断过曝）
        
        # 放大物理特征的权重，使其在余弦相似度计算中占据主导
        features.extend([l_mean * 2, l_std * 2, l_p10 * 2, l_p90 * 2])

        # ==========================================
        # 3. 锐度特征 (Laplacian 方差)
        # ==========================================
        # 量化图像的清晰/模糊程度 (尺寸: 1 维)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 归一化到一个相对合理的区间 (经验：模糊图通常<100，清晰图通常>500)
        sharpness = min(laplacian_var / 1000.0, 1.0) 
        features.append(sharpness * 3) # 锐度在图像增强中非常重要，赋予高权重

        # ==========================================
        # 4. 纹理与噪声代理特征 (边缘密度 & 高频残差)
        # ==========================================
        # 捕捉图像中的噪点颗粒和细节密集程度 (尺寸: 2 维)
        
        # 4.1 Canny 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (256 * 256)
        
        # 4.2 高频残差 (原图减去平滑图的绝对值平均)
        # 噪点极高的图片，残差均值会很大
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_residual = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
        noise_level = np.mean(noise_residual) / 255.0
        
        features.extend([edge_density * 2, noise_level * 3])

        # ==========================================
        # 5. 向量归一化
        # ==========================================
        vector = np.array(features)
        # 防止除零错误
        vector = vector / (np.linalg.norm(vector) + 1e-8) 

        return vector.tolist()

    def _get_prompt_embedding(self, prompt: str) -> list[float]:
        """
        提取自然语言意图特征。
        此处提供伪实现，生产环境中请使用 SentenceTransformer 或 OpenAI 的 text-embedding 模型。
        """
        return self.text_model.encode(prompt).tolist()
        # 伪向量占位 (假设 256 维)
        # return np.random.rand(256).tolist() 

    def _get_combined_embedding(self, img: np.ndarray, prompt: str) -> list[float]:
        """将图像特征与文本特征拼接，形成多模态联合向量 (The Joint Key)"""
        img_emb = self._get_image_embedding(img)
        prompt_emb = self._get_prompt_embedding(prompt)
        # 拼接并归一化
        combined = np.concatenate([img_emb, prompt_emb])
        combined = combined / np.linalg.norm(combined)
        return combined.tolist()

    def add_experience(self, img: np.ndarray, prompt: str, code_str: str, best_params: dict, score: float):
        """将成功的处理经验入库"""
        embedding = self._get_combined_embedding(img, prompt)
        doc_id = str(uuid.uuid4())
        
        # 将结构化数据序列化为 JSON 字符串存入 Document，也可以拆解存入 Metadata
        payload = {
            "code_str": code_str,
            "best_params": best_params,
            "score": score
        }
        
        self.collection.add(
            embeddings=[embedding],
            documents=[json.dumps(payload)],
            metadatas=[{"prompt": prompt, "score": score}], # 元数据可用于辅助过滤
            ids=[doc_id]
        )
        print(f"✅ 经验已入库 (ID: {doc_id}, Score: {score:.2f})")

    def query_experience(self, img: np.ndarray, prompt: str, top_k: int = 1) -> dict | None:
        """检索最相似的历史经验"""
        if self.collection.count() == 0:
            return None

        embedding = self._get_combined_embedding(img, prompt)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        
        if not results['documents'][0]:
            return None

        # 解析命中结果
        best_match_str = results['documents'][0][0]
        distance = results['distances'][0][0] # 因为是 cosine，越小越相似 (通常 0.0 代表完全一致)
        
        # 将 Cosine Distance 转换为相似度分数 (0~1)
        similarity = 1.0 - distance 
        
        payload = json.loads(best_match_str)
        payload["similarity"] = similarity
        
        return payload