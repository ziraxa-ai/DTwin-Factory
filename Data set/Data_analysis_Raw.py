import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
cars = pd.read_csv("out_cars.csv")
orders = pd.read_csv("out_orders.csv")
shipments = pd.read_csv("out_shipments.csv")

# -------------------------
# 1. Cars Analysis
# -------------------------
plt.figure(figsize=(8,5))
cars['model'].value_counts().plot(kind='bar')
plt.title('Number of Cars Produced by Model')
plt.xlabel('Model')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('cars_by_model.png')

plt.figure(figsize=(8,5))
quality_rate = cars['quality_pass'].mean()*100
plt.bar(['Pass','Fail'], [quality_rate, 100-quality_rate], color=['green','red'])
plt.title('Quality Pass Rate (%)')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('cars_quality_rate.png')

plt.figure(figsize=(8,5))
plt.hist(cars['lead_time_sec']/60, bins=20, color='skyblue', edgecolor='black')
plt.title('Lead Time Distribution (minutes)')
plt.xlabel('Lead time (min)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('cars_lead_time.png')

# -------------------------
# 2. Orders Analysis
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(orders['ts'], orders['qty'], marker='o')
plt.title('Orders Quantity Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Quantity')
plt.tight_layout()
plt.savefig('orders_quantity_time.png')

plt.figure(figsize=(8,5))
plt.hist(orders['price'], bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Order Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('orders_price_distribution.png')

plt.figure(figsize=(8,5))
plt.scatter(orders['qty'], orders['price'], alpha=0.7)
plt.title('Order Quantity vs Price')
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.tight_layout()
plt.savefig('orders_qty_vs_price.png')

# -------------------------
# 3. Shipments Analysis
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(shipments['ts'], shipments['qty'], marker='x')
plt.title('Shipments Quantity Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Quantity')
plt.tight_layout()
plt.savefig('shipments_quantity_time.png')

plt.figure(figsize=(8,5))
plt.hist(shipments['travel_time']/3600, bins=10, color='purple', edgecolor='black')
plt.title('Shipment Travel Time Distribution (hours)')
plt.xlabel('Travel time (hours)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('shipments_travel_time.png')

plt.figure(figsize=(8,5))
route_counts = shipments['route'].value_counts().head(10)
route_counts.plot(kind='barh')
plt.title('Top Shipment Routes')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('shipments_routes.png')

print("Analysis complete. Figures saved as PNG files.")
