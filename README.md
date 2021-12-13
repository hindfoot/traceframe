
# Tracesframe

Tracesframe is a Python package to work with distributed
traces, particularly [Jaeger](https://www.jaegertracing.io/) distributed traces, and the [pandas](https://pandas.pydata.org/) tool.

It also uses the [HoloViews](https://holoviews.org/) library.

A tutorial showing the use of the library can be seen at https://github.com/hindfoot/jaeger-analytics-intro

## Model

We define a straw-man pandas.DataFrame schema for **traces** and **spans**.

### Traces

|  Column    | Dtype          |
|  ------    | -------------- |
|  traceID   | object         |
|  traceName | object         |
|  nspans    | int64          |
|  errspans  | int64          |
|  duration  | timedelta64[ns]|
|  startTime | datetime64[ns] |
|  processes | object         |
|  iserror   | bool           |

### Spans

Data columns (total 71 columns):
|  Column    | Dtype          |
|  ------    | -------------- |
|  traceID   | object         |
|  traceName | object         |
|  nspans    | int64          |
|  errspans  | int64          |
|  duration  | timedelta64[ns]|
|  startTime | datetime64[ns] |
|  processes | object         |
|  iserror   | bool           |


| Column                       | Dtype  |
| ------                       | -----  |
| traceID                      | object |
| spanID                       | object |
| flags                        | int64  |
| operationName                | object |
| startTime                    | int64  |
| duration                     | int64  |
| logs                         | object |
| processID                    | object |
| service                      | object |
| parent                       | object |
| &lt;tag&gt;                  | object |

## Testing

`python -m unittest`

