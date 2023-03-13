# recipe_extractor
Uses BERT to extract the ingredients and recipe from a cooking website.
The project is built in such a way as to allow future extensions and pipeline composition.

### Scrap the data
First, the program needs to create a dataset of all the paragraphs available.
Place the provided intial csv at the project root and un the scrapper:

    cd scrapper
    python loveandlemons.py
    
That will create a csv with all paragraphs from the given urls, 
and indicate if the paragraph contains the ingredients, recipe, or just filler text

### Tokenize for BERT
For the model to run we need to tokenize the scrapped paragraphs. 
Simply feed the csv from the scapper to the tokenized dataset module:

    cd datasets
    tokenized_dataset.py
    
### Train
We are now ready to train a classifier. 
The classifier will try to predict if a text is a recipe, ingredients or filler text.
Since the data is highly unbalanced (most of the paragraphs are filler),
the training scripts balances the data by oversampling the minority classes.

From the project root:

    python train_model.py
    
### api
A simple api is provided. The api takes in a url and returns the ingredients and recipe within the given website.
To start the server:

    cd api
    flask --app server.py run
    
To query, send a POST request to API/recipe with the url inside the body. For example:
    
    curl -i -X POST -H 'Content-Type: application/json' -d "{\"url\": \"<YOUR URL>\"}" http://<API ADDRESS>>/recipe



    
    
