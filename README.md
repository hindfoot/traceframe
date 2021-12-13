
# Tracesframe

Tracesframe is a proof-of-concept Python package to work with distributed
traces, particularly [Jaeger](https://www.jaegertracing.io/) distributed traces, and the [pandas](https://pandas.pydata.org/) tool.

It also uses the [HoloViews](https://holoviews.org/) library.

A tutorial Jupyter notebook showing the use of the library can be seen at https://github.com/hindfoot/jaeger-analytics-intro


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

## Functions

- Jaeger API to Python or pandas structures
  - known_services(endpoint, password)
  - traces_from_jaeger(endpoint, password, limit, service, op, tag_expr, …) 
  - spans_from_jaeger(endpoint, password, limit, service, op, tag_expr, …)
- Elasticsearch to the same structures (experimental)
  - traces_from_es()
  - …
- Analysis
  - get_critical_segments([]spans)
- Visualization
  - pretty_trace_table()
  - showSingleTrace()
  - …

## Development

### Testing

```bash
cd test
python -m unittest
```

