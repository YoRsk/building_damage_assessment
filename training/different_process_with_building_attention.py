"""
使用扩展边界框获取建筑物周围的上下文预测和概率
分析上下文区域中非零类别的分布
结合上下文信息和建筑物自身的第二高概率类别来做决策：

如果第二高概率类别与上下文主导类别相同，直接使用该类别
如果不同，比较两个类别的概率，选择概率更高的类别
保持了原有的小型建筑物判断逻辑，但增加了上下文信息的考虑

目前结果，最小的建筑还是有问题
"""
def process_with_building_attention(self, prediction_prob, building_mask):
    """
    Process small buildings with special attention, considering surrounding context when unclassified
    Args:
        prediction_prob: softmax probability distribution (H, W, 5)
        building_mask: building mask
    """
    # Get initial prediction using argmax
    prediction = np.argmax(pvrediction_prob, axis=2)
    
    if prediction.shape != building_mask.shape:
        raise ValueError("Prediction and building_mask must have the same shape")
            
    enhanced_pred = prediction.copy()
    building_labels = measure.label(building_mask)
    
    print("\nProcessing small buildings...")
    for building_id in tqdm(range(1, building_labels.max() + 1), desc="Analyzing small buildings"):
        curr_building_mask = building_labels == building_id
        building_size = np.sum(curr_building_mask)
        
        if building_size < self.config['SMALL_BUILDING_THRESHOLD']:
            # Get building boundary box
            props = measure.regionprops(curr_building_mask.astype(int))
            bbox = props[0].bbox
            
            # Extend receptive field
            y1, x1, y2, x2 = bbox
            pad = self.config['CONTEXT_WINDOW'] // 2
            y1_ext = max(0, y1 - pad)
            x1_ext = max(0, x1 - pad)
            y2_ext = min(prediction.shape[0], y2 + pad)
            x2_ext = min(prediction.shape[1], x2 + pad)
            
            # Get predictions for the building and its context
            current_pred = prediction[curr_building_mask]
            context_pred = prediction[y1_ext:y2_ext, x1_ext:x2_ext]
            current_probs = prediction_prob[curr_building_mask]
            context_probs = prediction_prob[y1_ext:y2_ext, x1_ext:x2_ext]
            
            # Only process unclassified buildings
            if np.all(current_pred == 0):
                # Consider both building probabilities and context
                context_classes = context_pred[context_pred > 0]  # Get non-zero classes in context
                
                if len(context_classes) > 0:
                    # Get distribution of classes in context
                    context_class_dist = np.bincount(context_classes)
                    if len(context_class_dist) > 1:  # If there are non-zero classes
                        dominant_context_class = np.argmax(context_class_dist[1:]) + 1
                        
                        # Get second highest probabilities for the building
                        sorted_indices = np.argsort(current_probs, axis=1)
                        second_best_classes = sorted_indices[:, -2]
                        
                        if len(second_best_classes) > 0:
                            main_second_class = np.bincount(second_best_classes).argmax()
                            
                            # If second-best class matches dominant context class, use it
                            if main_second_class == dominant_context_class:
                                enhanced_pred[curr_building_mask] = main_second_class
                            # Otherwise, use the class with higher probability
                            elif main_second_class > 0:
                                # Compare probabilities
                                second_best_prob = np.mean(current_probs[:, main_second_class])
                                context_class_prob = np.mean(context_probs[..., dominant_context_class])
                                
                                if second_best_prob > context_class_prob:
                                    enhanced_pred[curr_building_mask] = main_second_class
                                else:
                                    enhanced_pred[curr_building_mask] = dominant_context_class
                            
    return enhanced_pred
    def process_with_building_attention(self, prediction_prob, building_mask):
        """
        对小型建筑物进行特殊处理，当预测为unclassified时选择第二可能的类别
        Args:
            prediction_prob: softmax输出的概率分布 (H, W, 5)
            building_mask: 建筑物掩码
        """
        # 先获取argmax的预测结果
        prediction = np.argmax(prediction_prob, axis=2)
        
        if prediction.shape != building_mask.shape:
            raise ValueError("Prediction and building_mask must have the same shape")
                
        enhanced_pred = prediction.copy()
        building_labels = measure.label(building_mask)
        
        print("\n处理小型建筑物...")
        for building_id in tqdm(range(1, building_labels.max() + 1), desc="分析小型建筑"):
            curr_building_mask = building_labels == building_id
            building_size = np.sum(curr_building_mask)
            
            if building_size < self.config['SMALL_BUILDING_THRESHOLD']:
                # 获取建筑物的边界框
                props = measure.regionprops(curr_building_mask.astype(int))
                bbox = props[0].bbox
                
                # 扩展感受野
                y1, x1, y2, x2 = bbox
                pad = self.config['CONTEXT_WINDOW'] // 2
                y1_ext = max(0, y1 - pad)
                x1_ext = max(0, x1 - pad)
                y2_ext = min(prediction.shape[0], y2 + pad)
                x2_ext = min(prediction.shape[1], x2 + pad)
                
                # 获取当前建筑物的预测和概率
                current_pred = prediction[curr_building_mask]
                current_probs = prediction_prob[curr_building_mask]
                
                # 只处理未分类的建筑物
                if np.all(current_pred == 0):
                    # 获取每个像素第二高概率的类别
                    sorted_indices = np.argsort(current_probs, axis=1)  # 按概率排序
                    second_best_classes = sorted_indices[:, -2]  # 获取第二高概率的类别
                    
                    if len(second_best_classes) > 0:
                        # 获取最常见的第二可能类别
                        main_second_class = np.bincount(second_best_classes).argmax()
                        if main_second_class > 0:  # 如果不是未分类
                            enhanced_pred[curr_building_mask] = main_second_class
                        
        return enhanced_pred