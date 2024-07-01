import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Read the already sampled data used for EDA
final_data=pd.read_csv('final_data_15.csv', encoding="ISO-8859-1")

# Get the unique users and movie IDs
unique_users = final_data['CustomerID'].unique()
unique_movies = final_data['MovieID'].unique()

# Print the number of unique users and movie IDs
print(f"Number of unique users: {len(unique_users)}")
print(f"Number of unique movies: {len(unique_movies)}")

# Preparing data for recommendation system
# Creating Index Mappings: user_to_index and movie_to_index are dictionaries that map each unique user and movie to a unique integer index.
# The process of index mapping is used to convert these IDs into a continuous range of integers starting from 0 
user_to_index = {user: idx for idx, user in enumerate(unique_users)}
movie_to_index = {movie: idx for idx, movie in enumerate(unique_movies)}

final_data['CustomerID'] = final_data['CustomerID'].map(user_to_index)
final_data['MovieID'] = final_data['MovieID'].map(movie_to_index)

# Split the data into training and validation sets
train_data, val_data = train_test_split(final_data, test_size=0.2, random_state=42)

# Create a custom Netflix dataset
class NetflixDataset(Dataset):
    def __init__(self, final_data):
        self.users = torch.tensor(final_data['CustomerID'].values, dtype=torch.float64)
        self.movies = torch.tensor(final_data['MovieID'].values, dtype=torch.int64)
        self.ratings = torch.tensor(final_data['Rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]
    
# Create Train and Validation DataLoaders
train_dataset = NetflixDataset(train_data)
val_dataset = NetflixDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the Neighborhood Model
class NeighborhoodModel(nn.Module):
    def __init__(self, num_users, num_movies):
        super(NeighborhoodModel, self).__init__()
        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, movie):
        user_bias = self.user_biases(user)
        movie_bias = self.movie_biases(movie)
        prediction = self.global_bias + user_bias + movie_bias
        return prediction.squeeze()

# Initialize the Neighborhood Model
model = NeighborhoodModel(len(unique_users), len(unique_movies))
model = model.to(device)
    
# Initialize the model parameters, loss function, and optimizer
num_factors = 200
alpha = 0.01
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Compute the Validation Metric: Mean Average Precision @ k(The number of predicted items to consider)
# Can compute the following for k = 5 or 10 values: Considering the top 5 or 10 recommendations

# Function average_precision_at_k returns: The average precision at k.
def average_precision_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

# Function mapk returns: The mean average precision at k.
def mapk(actual, predicted, k=10): #k=5
    map_values = []
    for user, items in predicted.items():
        if user not in actual or not items:
            continue
        else:
            ap = average_precision_at_k(actual[user], items, k)
            map_values.append(ap)
    return sum(map_values) / len(map_values)

# Model Training
num_epochs = 5

# Lists to store training and validation losses for each epoch
train_losses = []
val_losses = []

# Lists to store validation metrics for each epoch
val_mapks = []
val_rmses = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for user, movie, rating in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        optimizer.zero_grad()
        user = user.long().to(device)
        movie = movie.long().to(device)
        rating = rating.float().view(-1, 1).to(device)
        prediction = model(user, movie).view(-1,1)
        loss = criterion(prediction, rating)
        loss.backward()
        optimizer.step()
        train_loss += sqrt(loss.item())

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_mapk = 0.0
    num_users = 0
    mov_usr = dict()
    with torch.no_grad():
        for user, movie, rating in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            for i, us in enumerate(user):
                u, m, r = us.item(), movie[i].item(), rating[i].item()
                if u not in mov_usr and r>2:
                    mov_usr[u] = [m]
                elif r>2:
                    mov_usr[u].append(m)

    with torch.no_grad():
        for user, movie, rating in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            user = user.long().to(device)
            movie = movie.long().to(device)
            rating = rating.float().view(-1, 1).to(device)

            prediction = model(user, movie)
            loss = criterion(prediction, rating)

            val_loss += sqrt(loss.item())

            # Calculate map@k
            actual_ratings = rating.cpu().numpy()
            predicted_ratings = prediction.cpu().numpy()
            pred_dict = dict()
            for i, pr in enumerate(predicted_ratings):
                if pr>2: 
                    us, mo = user[i].item(), movie[i].item()
                    if us in pred_dict:
                        pred_dict[us].append(mo)
                    else:
                        pred_dict[us] = [mo]
            val_mapk += mapk(mov_usr, pred_dict, k=10) #k=5
            num_users += 1 #user.shape[0]
            
    val_loss /= len(val_loader)
    val_mapk /= num_users
    val_rmse = sqrt(val_loss)
    val_losses.append(val_loss)
    val_mapks.append(val_mapk)
    val_rmses.append(val_rmse)

    
    train_losses.append(train_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")#", Train MAP@k: {train_mapk:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}, Validation MAP@10: {val_mapk:.4f}")
 
# Storing the train loss, validation loss, valtion rmse and validation map@k values for the model    
neigh_10={"Train Loss:": train_losses, "Validation Loss": val_losses, "Validation RMSE": val_rmses, "Validation MAP@5": val_mapks}
print(neigh_10)

# Visualization of the above for generated parameters
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='#006400')  # Dark green
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='#32CD32')
plt.xscale('log')  # Setting log scale for x-axis
plt.xlabel('Epoch (log scale)')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs (Log Scale)')
plt.legend()
plt.savefig('training_validation_loss_neigh10.png')
plt.show()

# Plotting Validation Metrics with log scale on x-axis
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), val_rmses, label='Validation RMSE', color='#8FBC8F')  # Dark sea green
plt.plot(range(1, num_epochs + 1), val_mapks, label='Validation MAP@10', color='#98FB98')  # Pale green
plt.xscale('log')  # Setting log scale for x-axis
plt.xlabel('Epoch (log scale)')
plt.ylabel('Metric Value')
plt.title('Validation Metrics Over Epochs (Log Scale)')
plt.legend()
plt.savefig('validation_metrics_neigh10.png')
plt.show()


# Inference: Get the Top Recommendations for a particular user
user_id = 72153  # recommendations for the given user
all_movie_ids = range(len(unique_movies)) 

# Convert to PyTorch tensors
user_tensor = torch.tensor([user_id] * len(all_movie_ids)).to(device)
movies_tensor = torch.tensor(all_movie_ids).to(device)

# Predict ratings
model.eval()
with torch.no_grad():
    predictions = model(user_tensor, movies_tensor).cpu().numpy()

# Sort the predictions such that IDs of movies with the highest predicted ratings
recommended_movie_ids = np.argsort(-predictions)  

# Filter out the alreday watched movies by the user
watched_movies = set(final_data[final_data['CustomerID'] == user_id]['MovieID'])
recommended_movie_ids = [mid for mid in recommended_movie_ids if mid not in watched_movies]

# Generate top N recommendations
print("Movies the user has already watched: ", watched_movies)
top_n_recommendations = recommended_movie_ids[:10]  # Top 10 recommendations
print("\nTop 10 recommendations for the user based on his watched history are: ", top_n_recommendations)