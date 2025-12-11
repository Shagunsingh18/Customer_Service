import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
data_path = "data.csv"              # <<---- IMPORTANT
data = pd.read_csv(data_path)

# Combine fields
data["ticket_text"] = (data["subject"].fillna("") + " " + data["body"].fillna(""))

# Category mapping
mapping = {
    "Technical Support": "Technical",
    "Billing and Payments": "Billing",
    "Returns and Exchanges": "General",
    "Sales and Pre-Sales": "Feedback",
    "Account Management": "Account"
}
data["category"] = data["queue"].map(mapping)

# Keep English
data = data[data["language"] == "en"].reset_index(drop=True)

# Select columns
data = data[[
    "ticket_text", "priority", "queue", "category",
    "tag_1", "tag_2", "tag_3"
]]

data.rename(columns={
    "tag_1": "main_tag",
    "tag_2": "sub_tag",
    "tag_3": "extra_tag"
}, inplace=True)

# ------------------------------------------------
# 2. INPUT / OUTPUT
# ------------------------------------------------
X = data[["ticket_text","priority","queue","main_tag","sub_tag","extra_tag"]]
y = data["category"]

categorical_cols = ["priority","queue","main_tag","sub_tag","extra_tag"]

# Label Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ------------------------------------------------
# 3. FEATURE EXTRACTION
# ------------------------------------------------

# TEXT (reduced TF-IDF to avoid overfitting)
tfidf = TfidfVectorizer(max_features=2000)
X_train_text = tfidf.fit_transform(X_train["ticket_text"])
X_test_text = tfidf.transform(X_test["ticket_text"])

# CATEGORICAL
ohe = OneHotEncoder(handle_unknown="ignore")
X_train_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_cat = ohe.transform(X_test[categorical_cols])

# COMBINE SPARSE MATRIX (IMPORTANT)
X_train_final = hstack([X_train_text, X_train_cat])
X_test_final = hstack([X_test_text, X_test_cat])

# Dense only for ANN
X_train_dense = X_train_final.toarray()
X_test_dense = X_test_final.toarray()

# ------------------------------------------------
# 4. TRAIN MODELS
# ------------------------------------------------

# -------- SVM (best for this task) --------
svm_model = LinearSVC()
svm_model.fit(X_train_final, y_train)
SVM_ACC = accuracy_score(y_test, svm_model.predict(X_test_final))

# -------- Logistic Regression --------
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_final, y_train)
LOG_ACC = accuracy_score(y_test, log_reg.predict(X_test_final))

# -------- ANN (improved) --------
class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.35)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

input_size = X_train_dense.shape[1]
output_size = len(le.classes_)

model = ANN(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_t = torch.tensor(X_train_dense, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

EPOCHS = 20
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

# ANN accuracy
X_test_t = torch.tensor(X_test_dense, dtype=torch.float32)
with torch.no_grad():
    preds = torch.argmax(model(X_test_t), dim=1).numpy()
NN_ACC = accuracy_score(y_test, preds)

# ------------------------------------------------
# 5. PREPROCESS FUNCTION (USED BY STREAMLIT)
# ------------------------------------------------
def preprocess(text, priority, queue, main_tag, sub_tag, extra_tag):
    txt_vec = tfidf.transform([text])
    cat_vec = ohe.transform([[priority, queue, main_tag, sub_tag, extra_tag]])
    combined = hstack([txt_vec, cat_vec]).toarray()  
    return combined

# ------------------------------------------------
# 6. ANSWER LOOKUP FOR RESPONSE
# ------------------------------------------------
raw_full = pd.read_csv(data_path)
raw_full["category"] = raw_full["queue"].map(mapping)

answer_map = {}
for cat in le.classes_:
    try:
        ans = raw_full[raw_full["category"] == cat]["answer"].dropna().iloc[0]
        answer_map[cat] = ans
    except:
        answer_map[cat] = "No predefined answer available."
