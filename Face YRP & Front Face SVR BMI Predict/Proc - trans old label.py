import pandas as pd

file_path = 'old_label.csv'
raw_df = pd.read_csv(file_path)

df = pd.DataFrame(columns=['id', 'height', 'weight', 'bmi-gt', 'class-gt'])


def feet_to_meter(feet_str: str):
    tokens = feet_str.split("'")
    feet = int(tokens[0].strip())

    tokens = tokens[1].split('"')
    inch = int(tokens[0].strip())

    cm = (feet * 30.48) + (inch * 2.54)

    return round(cm / 100, 2)


def lb_to_kg(pound: str):
    return round(int(pound) / 2.20462262, 2)


def get_bmi(weight, height):
    return round(weight / (height ** 2), 2)


# paper 分法
def get_bmi_class(bmi):
    if bmi < 18.5:
        return 'a'
    elif 18.5 <= bmi < 25:
        return 'b'
    elif 25 <= bmi < 30:
        return 'c'
    else:
        return 'd'


for index, row in raw_df.iterrows():
    data = list()
    data.append(str(row['ID']) + '.jpg')

    meter = feet_to_meter(row['Height'])
    data.append(meter)

    kg = lb_to_kg(row['Weight'])
    data.append(kg)

    bmi_value = get_bmi(kg, meter)
    data.append(bmi_value)

    bmi_class = get_bmi_class(bmi_value)
    data.append(bmi_class)

    df.loc[len(df)] = data

    # break

df.to_csv('bmi_map.csv', encoding='utf-8', index=False)
