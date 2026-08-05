# Raw and Interim Data Dictionary

Detailed description of core transaction and identity tables in the IEEE-CIS Fraud Detection dataset.

---

## 💳 Transaction Table

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `TransactionID` | Integer | Unique identifier for each transaction (Primary Key). |
| `isFraud` | Integer (0/1) | Target variable: `1` indicates fraudulent transaction, `0` indicates legitimate. |
| `TransactionDT` | Float/Integer | Timedelta in seconds from reference epoch. |
| `TransactionAmt` | Float | Transaction payment amount in USD. |
| `ProductCD` | String | Product code prefix for transaction category (e.g. W, C, R, H, S). |
| `card1` - `card6` | Categorical / Int | Payment card information (Card issuer, card type, card category). |
| `addr1`, `addr2` | Categorical / Int | Billing address region and country codes. |
| `dist1`, `dist2` | Float | Distance metrics between transaction location and billing address. |
| `P_emaildomain` | String | Purchaser email domain (e.g. gmail.com, yahoo.com). |
| `R_emaildomain` | String | Recipient email domain. |
| `C1` - `C14` | Float / Int | Counting features (e.g. number of addresses associated with payment card). |
| `D1` - `D15` | Float | Timedelta features (e.g. days since previous transaction). |
| `M1` - `M9` | Categorical | Match features (e.g. name on card matches address). |
| `V1` - `V339` | Float | Vesta engineered features (ranking, counts, associations). |

---

## 🆔 Identity Table

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `TransactionID` | Integer | Foreign key referencing `TransactionTable.TransactionID`. |
| `id_01` - `id_38` | Mixed | Identity attributes (device info, IP hash, screen resolution, browser). |
| `DeviceType` | String | Operating device category (e.g. `mobile`, `desktop`). |
| `DeviceInfo` | String | Device brand and build string (e.g. `iOS Device`, `Windows`). |
