from flask import Flask, render_template, request, url_for, redirect, session
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secret key for session management
# Load your original dataset (replace 'new_data.csv' with your actual file path)
df = pd.read_csv('new_data.csv')
#wishlist_df = pd.read_csv('wishlist.csv')
orders_df = pd.read_csv('orders.csv')
wishlist_df = pd.read_csv('wishlist.csv')
df['category'] = df['category'].str.strip()
df['brand'] = df['brand'].str.strip()

# Load your products dataset (replace 'products.csv' with your actual file path)
products_df = pd.read_csv('products.csv')

# Preprocessing and model initialization
encoder = OneHotEncoder(sparse=False)
scaler = StandardScaler()

encoded_categories = encoder.fit_transform(df[['category', 'brand', 'gender', 'sub_cat']])
scaled_features = scaler.fit_transform(df[['price', 'rating']])
#scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the selected features
#scaled_features = scaler.fit_transform(df[['price', 'rating']])

encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(['category', 'brand', 'gender', 'sub_cat']))
encoded_df[['price', 'rating']] = scaled_features
encoded_df['f_assured'] = df['f_assured']

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        id = request.form['id']
        print(id)
        password = request.form['password']
        session['user_id'] = id

        users_df = pd.read_csv('users.csv')
        matched_user = users_df[(users_df['id'] == id) & (users_df['password'] == int(password))]

        if not matched_user.empty:
            user_name = matched_user.iloc[0]['user_name']  # Fetch user_name from the matched user
            session['user_name'] = user_name
            return redirect(url_for('choose_gender'))

    return render_template('login1.html')




@app.route('/wishlist')
def wishlist():
    user_id = session.get('user_id', '')  # Get the user's ID from the session

    if user_id:
        # Retrieve the user's wishlist based on their ID
        user_wishlist = wishlist_df[wishlist_df['users'] == user_id]

        if not user_wishlist.empty:
            product_indices = user_wishlist['products'].iloc[0]
            product_indices = eval(product_indices)  # Convert the string to a list

            # Fetch the product details for the wishlist items
            wishlist_items = []
            for idx in product_indices:
                product_data = products_df.iloc[idx]
                wishlist_items.append(product_data)

            return render_template('wishlist.html', wishlist_items=wishlist_items)

    # Redirect to login or handle if user is not logged in or has no wishlist
    return redirect(url_for('login'))

# ... (rest of the code)


@app.route('/choose_gender', methods=['GET', 'POST'])
def choose_gender():
    user_id = session.get('user_name', '')
    if request.method == 'POST':
        gender = request.form['gender']
        session['gender'] = gender
        print(gender)
        return redirect(url_for('choose_category'))

    return render_template('choose_gender.html',user_id=user_id)

@app.route('/my_orders')
def my_orders():
    user_id = session.get('user_id', '')

    if user_id:
        # Read the orders.csv file
        orders_df = pd.read_csv('orders.csv')

        # Filter user orders based on user_id
        user_orders = orders_df[orders_df['users'] == user_id]

        # Get the product indexes from the user's orders
        product_indexes = user_orders['products'].apply(eval).explode()

        # Retrieve the corresponding product data from products_df
        ordered_products = products_df.loc[product_indexes]

        return render_template('my_orders.html', ordered_products=ordered_products,)

    return redirect(url_for('login1'))



@app.route('/choose_category', methods=['GET', 'POST'])
def choose_category():
    user_id = session.get('user_name', '')
    all_indices = []
    for index_list in orders_df['products']:
        all_indices.extend(eval(index_list))
    for index_list in wishlist_df['products']:
        all_indices.extend(eval(index_list))

    # Find most common indices
    most_common_indices = [index for index, count in Counter(all_indices).most_common()]

    # Get the corresponding product details for the most common indices
    most_common_products = [products_df.iloc[index] for index in most_common_indices]

    if request.method == 'POST':
        category = request.form['category']
        session['category'] = category
        return redirect(url_for('index'))
    return render_template('choose_category.html', user_id=user_id, popular_items=most_common_products)
#category = session.get('category', '').lower()
#gender = session.get('gender', '').lower()  # Get stored gender from session


@app.route('/index')
def index():
    category = session.get('category', '').lower()
    gender = session.get('gender', '').lower()  # Get stored gender from session

    # Filter rows based on input category and gender
    filtered_rows = df[(df['category'] == category) & (df['gender'] == gender)]
    filtered_indices = filtered_rows.index
    print(len(filtered_rows))

    # Reset indices of encoded_df to match filtered_indices
    encoded_df_filtered = encoded_df.loc[filtered_indices]

    # Initialize the Nearest Neighbors model
    knn = NearestNeighbors(n_neighbors=len(filtered_indices))
    knn.fit(encoded_df_filtered)

    # Use the session user_id to fetch user orders
    user_id = session.get('user_id', '')
    if user_id:

        # Filter user orders based on user_id
        user_orders = orders_df[orders_df['users'] == user_id]
        wish_orders = wishlist_df[wishlist_df['users'] == user_id]

        # Convert the string representations of lists to actual lists
        user_orders['price'] = user_orders['price'].apply(eval)
        user_orders['rating'] = user_orders['rating'].apply(eval)
        user_orders['f_assured'] = user_orders['f_assured'].apply(eval)
        wish_orders['price'] = wish_orders['price'].apply(eval)
        wish_orders['rating'] = wish_orders['rating'].apply(eval)
        wish_orders['f_assured'] = wish_orders['f_assured'].apply(eval)

        arry = []

        # Calculate average price from the list of prices
        avg_price1 = user_orders['price'].apply(np.mean)
        #avg_price = user_orders['price'].apply(np.mean).mean()
        avg_price2 = wish_orders['price'].apply(np.mean)
        avg_price = np.mean([avg_price1.mean(), avg_price2.mean()])
        print('price' , float(avg_price))

        # Find the most common brand in the brand columns
        brand_counts1 = user_orders['brand'].apply(eval).explode().value_counts()
        brand_counts2 = wish_orders['brand'].apply(eval).explode().value_counts()
        brand_counts = brand_counts1.add(brand_counts2, fill_value=0)
        most_common_brand = brand_counts.idxmax()
        arry.append(float(brand_counts.max()))
        #print("brand",float(brand_counts.max()))

        #Find the most common rating in the rating columns
        #most_common_rating = user_orders['rating'].apply(np.mean).mean()
        rating_counts1 = user_orders['rating'].explode().value_counts()
        rating_counts2 = wish_orders['rating'].explode().value_counts()
        rating_counts = rating_counts1.add(rating_counts2, fill_value=0)
        most_common_rating = rating_counts.idxmax()
        arry.append(float(rating_counts.max()))
        print("rating", float(rating_counts.max()))


        # Find the most common sub_cat in the sub_cat column
        sub_cat_counts1 = user_orders['sub_cat'].apply(eval).explode().value_counts()
        sub_cat_counts2 = wish_orders['sub_cat'].apply(eval).explode().value_counts()
        sub_cat_counts = sub_cat_counts1.add(sub_cat_counts2, fill_value=0)
        most_common_sub_cat = sub_cat_counts.idxmax()
        arry.append(float(sub_cat_counts.max()))
        #print("sub_cat", float(sub_cat_counts.max()))

        # Find the most common f_assured in the f_assured columns
        f_assured_counts1 = user_orders['f_assured'].explode().value_counts()
        f_assured_counts2 = wish_orders['f_assured'].explode().value_counts()
        f_assured_counts = f_assured_counts1.add(f_assured_counts2, fill_value=0)
        most_common_f_assured = f_assured_counts.idxmax()
        arry.append(float(f_assured_counts.max()))
        #print("f_assured",float(f_assured_counts.max()))

        print(arry,"array")
        total_orders = user_orders['price'].apply(len)
        total_wish = wish_orders['price'].apply(len)
        total = float(total_orders.sum() + total_wish.sum())
        print(total)

        arry = np.array(arry)
        for i in range(0,len(arry)):
            val = arry[i]/total
            if val > 0.6:
                arry[i] = 2
            elif val < 0.5:
                arry[i] = 0.5
            else:
                if i != 3:
                    arry[i] = 1.0
        print("f_assured", arry[3]/total)
        if most_common_f_assured == 0 and arry[3]/total >= 0.3 :
            arry[3] = 0.5

        print("new",arry)
        # Convert brand and sub_cat to lowercase
        brand = most_common_brand.lower()
        sub_cat = most_common_sub_cat.lower()
        print("array",avg_price)
        input_encoded = encoder.transform(np.array([category, brand, gender, sub_cat]).reshape(1, -1))
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['category', 'brand', 'gender', 'sub_cat']))
        input_scaled = scaler.transform(np.array([avg_price, most_common_rating]).reshape(1, -1))
        input_encoded_df[['price', 'rating']] = input_scaled
        input_encoded_df[['f_assured']] = most_common_f_assured

        input_encoded_df['brand_'+most_common_brand] *= arry[0]
        input_encoded_df['rating'] *= arry[1]
        input_encoded_df['sub_cat_' + most_common_sub_cat] *= arry[2]
        input_encoded_df['f_assured'] *= arry[3]
        input_encoded_df['price'] *= 2.0
        print(input_encoded_df['brand_'+most_common_brand],input_encoded_df['rating'],input_encoded_df['sub_cat_' + most_common_sub_cat],input_encoded_df['f_assured'],input_encoded_df['price'])
        # Filter rows based on input category and gender
        filtered_rows = df[(df['category'] == category) & (df['gender'] == gender)]
        filtered_indices = filtered_rows.index

        # Reset indices of encoded_df to match filtered_indices
        encoded_df_filtered = encoded_df.loc[filtered_indices]

        # Initialize the Nearest Neighbors model
        print(input_encoded_df)
        knn = NearestNeighbors(n_neighbors=len(filtered_indices))
        knn.fit(encoded_df_filtered)
        print(input_encoded_df['sub_cat_hs'])
        # Find the indices of the most similar rows to the input
        similar_distances, similar_indices = knn.kneighbors(input_encoded_df)

        # Sort indices based on distance (similarity)
        sorted_indices = similar_indices[0][similar_distances[0].argsort()]


        user_profile = input_encoded_df.iloc[0].values.reshape(1, -1)

        # Calculate cosine similarities between user profile and recommended item profiles
        cosine_similarities = cosine_similarity(user_profile, encoded_df_filtered.iloc[sorted_indices])

        # Calculate precision based on cosine similarities
        total_items = min(10, len(sorted_indices))  # In case there are fewer than 10 items
        similarity_sum = cosine_similarities[0][:total_items].sum()
        precision = similarity_sum / total_items

        # Print the precision value
        print("Precision:", precision)
        similar_products = []

        for idx in sorted_indices:
            product_name = products_df['products'].iloc[filtered_indices[idx]]
            image_url = products_df['image_url'].iloc[filtered_indices[idx]]
            brand = products_df['brand'].iloc[filtered_indices[idx]]
            price = products_df['price'].iloc[filtered_indices[idx]]
            f_assured = bool(products_df['f_assured'].iloc[filtered_indices[idx]])  # Convert to boolean
            rating = products_df['rating'].iloc[filtered_indices[idx]]

            product_data = {
                'product_name': product_name,
                'image_url': image_url,
                'brand': brand,
                'price': price,
                'f_assured': f_assured,
                'rating': rating
            }

            similar_products.append(product_data)


        return render_template('index.html', similar_products=similar_products)



if __name__ == '__main__':
    app.run(debug=True)



