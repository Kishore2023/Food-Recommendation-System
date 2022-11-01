import streamlit as st
import pickle
import pandas as pd

st.title('Food Recommendation')
# breakfast = pd.read_excel("D:\Project\Data\breakfast.xlsx")

food_list = pickle.load(open('Order1.pkl','rb'))
# food_list = food_list['DishName'].values
data = pd.DataFrame(food_list)
similarity = pickle.load(open('similarity.pkl','rb'))

Dining_list = pickle.load(open('DiningType.pkl','rb'))
DiningType = pd.DataFrame(Dining_list)

Variety_list = pickle.load(open('Variety.pkl','rb'))
Variety = pd.DataFrame(Variety_list)


def recommend(food):
    food_index = data[data["Order1"] == food].index[0]
    distances = similarity[food_index]
    food_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:11] 

    recommended_foods= []
    for i in food_list:
        recommended_foods.append(data.iloc[i[0]].Order1)
    return recommended_foods
    

meal_type = st.selectbox('Select Meal Type',('Select Meal Type','Breakfast','Lunch','Dinner') )
if meal_type is ('Breakfast'):
    food_break = DiningType.Breakfast
    for i in food_break:
        st.write(i)

if meal_type is ('Lunch'):
    food_desserts = DiningType.Lunch
    for i in food_desserts:
        st.write(i)
        
if meal_type is ('Dinner'):
    food_desserts = DiningType.Dinner
    for i in food_desserts:
        st.write(i)

veg = st.selectbox('Choose Veg or Non-Veg or Both', ('Select','Vegetarian','NonVegetarian', 'Both'))

if meal_type is ('Breakfast') and  veg is ('Vegetarian'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.Vegmain
    for i in food_veg:
        st.write(i)

if meal_type is ('Lunch') and  veg is ('Vegetarian'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.Vegmain
    for i in food_veg:
        st.write(i)

if meal_type is ('Dinner') and  veg is ('Vegetarian'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.Vegmain
    for i in food_veg:
        st.write(i)    


if meal_type is ('Breakfast') and  veg is ('NonVegetarian'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.NonVegmain
    for i in food_veg:
        st.write(i)

if meal_type is ('Lunch') and  veg is ('NonVegetarian'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.NonVegmain
    for i in food_veg:
        st.write(i)

if meal_type is ('Dinner') and  veg is ('NonVegetarian'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.NonVegmain
    for i in food_veg:
        st.write(i)    

if meal_type is ('Breakfast') and  veg is ('Both'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.Both
    for i in food_veg:
        st.write(i)
        
if meal_type is ('Lunch') and  veg is ('Both'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.Both
    for i in food_veg:
        st.write(i)

if meal_type is ('Dinner') and  veg is ('Both'):
    # if veg is ('Vegetarian'):
    food_veg = Variety.Both
    for i in food_veg:
        st.write(i)

selected_food = st.selectbox('What would you like to Order?',food_list)
if st.button('Recommend'):
    recommendations = recommend(selected_food)
    st.subheader("Also try this")
    for i in recommendations:
        st.write(i)