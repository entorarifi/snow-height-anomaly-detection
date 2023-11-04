key = "dm:columnorder"
json = {
    "select": 0,
    "tasks:id": 1,
    "tasks:data.station_code": 2,
    "tasks:data.start_date": 3,
    "tasks:data.end_date": 4,
    "tasks:data.number_of_datapoints": 5,
    "tasks:predictions_score": 6,
    "tasks:data.csv": 7,
    "tasks:total_annotations": 8,
    "tasks:cancelled_annotations": 9,
    "tasks:total_predictions": 10,
    "tasks:annotators": 11,
    "tasks:completed_at": 12,
    "show-source": 13
}

localStorage.setItem(key, JSON.stringify(json))
