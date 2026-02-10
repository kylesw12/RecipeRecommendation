# RecipeRecommendation

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kylesw12/RecipeRecommendation.git
   cd RecipeRecommendation

2. **Download the Datasets**:  
    Download the datasets from 
    https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=recipes.csv  
    After downloading, place the CSV files into a folder called data/ in the project root:  
    RecipeRecommendation/  
    ├── data/  
    │   ├── recipes.csv  
    │   └── reviews.csv  

3. **Run preprocessing and indexing**:  
    python indexing/preprocess.py  
    python indexing/indexer.py  
