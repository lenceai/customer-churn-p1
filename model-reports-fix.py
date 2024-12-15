def create_model_reports(self):
        """
        Generate and save classification reports and feature importance plots.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Verify models exist
            if self.rfc_model is None or self.lr_model is None:
                raise ValueError("Models not trained. Run train_models first.")

            # Get predictions
            y_train_preds_rf = self.rfc_model.predict(self.X_train)
            y_test_preds_rf = self.rfc_model.predict(self.X_test)
            y_train_preds_lr = self.lr_model.predict(self.X_train)
            y_test_preds_lr = self.lr_model.predict(self.X_test)
            
            # Random Forest Classification Report
            plt.figure(figsize=const.CLASSIFICATION_REPORT_FIG_SIZE)
            plt.text(0.01, 1.25, 'Random Forest Train', fontsize=10, fontproperties='monospace')
            plt.text(0.01, 0.05, classification_report(self.y_test, y_test_preds_rf), 
                    fontsize=10, fontproperties='monospace')
            plt.text(0.01, 0.6, 'Random Forest Test', fontsize=10, fontproperties='monospace')
            plt.text(0.01, 0.7, classification_report(self.y_train, y_train_preds_rf), 
                    fontsize=10, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(os.path.join(const.RESULTS_IMAGES_PATH, const.RFC_CLASSIFICATION_REPORT))
            plt.close()
            
            # Logistic Regression Classification Report
            plt.figure(figsize=const.CLASSIFICATION_REPORT_FIG_SIZE)
            plt.text(0.01, 1.25, 'Logistic Regression Train', fontsize=10, fontproperties='monospace')
            plt.text(0.01, 0.05, classification_report(self.y_train, y_train_preds_lr), 
                    fontsize=10, fontproperties='monospace')
            plt.text(0.01, 0.6, 'Logistic Regression Test', fontsize=10, fontproperties='monospace')
            plt.text(0.01, 0.7, classification_report(self.y_test, y_test_preds_lr), 
                    fontsize=10, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(os.path.join(const.RESULTS_IMAGES_PATH, const.LR_CLASSIFICATION_REPORT))
            plt.close()
            
            # Feature Importance Plot
            plt.figure(figsize=const.FEATURE_IMPORTANCE_FIG_SIZE)
            importances = self.rfc_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            names = [self.X_train.columns[i] for i in indices]
            plt.title("Feature Importance")
            plt.ylabel('Importance')
            plt.bar(range(self.X_train.shape[1]), importances[indices])
            plt.xticks(range(self.X_train.shape[1]), names, rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(const.RESULTS_IMAGES_PATH, const.FEATURE_IMPORTANCE_PLOT))
            plt.close()
            
            # ROC Curves
            plt.figure(figsize=const.PLT_FIGURE_SIZE)
            # Replace inf values with nan
            X_test_clean = self.X_test.replace([np.inf, -np.inf], np.nan)
            
            lrc_plot = RocCurveDisplay.from_estimator(
                self.lr_model, 
                X_test_clean, 
                self.y_test,
                name="Logistic Regression"
            )
            plt.title("ROC Curve - Logistic Regression")
            plt.savefig(os.path.join(const.RESULTS_IMAGES_PATH, const.LRC_ROC_CURVE))
            plt.close()
            
            # Combined ROC Curves
            fig, ax = plt.subplots(figsize=const.PLT_FIGURE_SIZE)
            RocCurveDisplay.from_estimator(
                self.rfc_model, 
                X_test_clean, 
                self.y_test,
                name="Random Forest",
                ax=ax
            )
            RocCurveDisplay.from_estimator(
                self.lr_model,
                X_test_clean,
                self.y_test,
                name="Logistic Regression",
                ax=ax
            )
            plt.title("ROC Curve Comparison")
            plt.savefig(os.path.join(const.RESULTS_IMAGES_PATH, const.ROC_CURVES_COMPARISON))
            plt.close()
            
            logging.info("Model reports created successfully")
            return True
            
        except Exception as err:
            logging.error("Creating model reports failed: %s", str(err))
            return False
