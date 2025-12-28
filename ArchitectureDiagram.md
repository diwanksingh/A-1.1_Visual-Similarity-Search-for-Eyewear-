flowchart TD

U[User<br/>(Web Browser)]
F[Next.js Frontend<br/>(Image Upload Screen)]
A[FastAPI Backend Server]
P[Image Preprocessing<br/>(Resize, Normalize, Clean)]
M[CLIP Vision Model<br/>(AI Brain)]
E[Embedding Vector<br/>(512 Numbers)]
V[FAISS Vector Index<br/>(101 Product Vectors)]
S[Similarity Ranking Engine]
D[Product Database<br/>(Name, Price, Brand, Images)]
R[Final Results<br/>(Top Matching Frames)]

U -->|Upload Image| F
F -->|POST /search| A
A --> P
P --> M
M --> E
E --> V
V --> S
S --> D
D --> R
