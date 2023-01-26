import tensorflow as tf
import cv2 as cv
MODEL_LABELS = ['American Samoa', 'Armenia', 'Bangladesh', 'Austria', 'Aland', 'Argentina', 'Andorra', 'Australia', 'Albania', 'Antarctica', 'Brazil', 'Belarus', 'Botswana', 'Belgium', 'Canada', 'Bhutan', 'Cambodia', 'Bulgaria', 'Bermuda', 'Bolivia', 'Croatia', 'Costa Rica', 'Czechia', 'Ecuador', 'Chile', 'Denmark', 'Dominican Republic', 'Colombia', 'Curacao', 'China', 'Estonia', 'Greece', 'Ghana', 'Finland', 'Germany', 'France', 'Egypt', 'Faroe Islands', 'Gibraltar', 'Eswatini', 'Iraq', 'Hong Kong', 'Guam', 'Guatemala', 'Hungary', 'India', 'Iceland', 'Greenland', 'Ireland', 'Indonesia', 'Laos', 'Jersey', 'Jordan', 'Isle of Man', 'Israel', 'Italy', 'Latvia', 'Kenya', 'Japan', 'Kyrgyzstan', 'Martinique', 'Malta', 'Lesotho', 'Mexico', 'Lebanon', 'Macao', 'Lithuania', 'Madagascar', 'Malaysia', 'Luxembourg', 'Myanmar', 'Mongolia', 'Monaco', 'Nepal', 'Nigeria', 'Mozambique', 'Netherlands', 'North Macedonia', 'New Zealand', 'Montenegro', 'Pakistan', 'Philippines', 'Peru', 'Palestine', 'Northern Mariana Islands', 'Pitcairn Islands', 'Portugal', 'Paraguay', 'Poland', 'Norway', 'Slovakia', 'Romania', 'Puerto Rico', 'Reunion', 'San Marino', 'Singapore', 'Serbia', 'Qatar', 'Russia', 'Senegal', 'Sri Lanka', 'Spain', 'South Sudan', 'Sweden', 'South Korea', 'South Georgia and South Sandwich Islands', 'South Africa', 'Slovenia', 'Switzerland', 'Svalbard and Jan Mayen', 'Tanzania', 'United Kingdom', 'Turkey', 'United Arab Emirates', 'Tunisia', 'Ukraine', 'Uganda', 'Taiwan', 'Thailand', 'United States', 'Venezuela', 'Uruguay', 'US Virgin Islands', 'Vietnam']


def get_top_3(output, labels):
  output_list = output[0]
  return sorted(zip(output_list, labels), reverse=True)[:3]
  

interpreter = tf.lite.Interpreter(
  model_path="eflite0_test1_geo.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv.imread("test1.png")
img = cv.resize(img, (224, 224))

interpreter.set_tensor(input_details[0]["index"], [img])
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"])
top = get_top_3(output_data, MODEL_LABELS)
print(f"Top predictions:")
for conf, pred in top:
  spacing = "".join([" " for _ in range(13-len(pred))])
  print(f"Country: {pred}{spacing}=== Confidence: {conf}%")
