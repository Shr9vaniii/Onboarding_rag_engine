## EXAMPLE 1
**SOURCE:** tests\test_tutorial\test_generate_clients\test_tutorial003.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_post_users
Arguments: []
Source_File: tests\test_tutorial\test_generate_clients\test_tutorial003.py
Docstring/Description:
```

---

## EXAMPLE 2
**SOURCE:** tests\test_security_http_base_optional.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_security_http_base
Arguments: []
Source_File: tests\test_security_http_base_optional.py
Docstring/Description:
```

---

## EXAMPLE 3
**SOURCE:** tests\test_ws_router.py

**CONTENT BLOCK:**
```
Type: function
 Name: middleware_constructor
Arguments: []
Source_File: tests\test_ws_router.py
Docstring/Description:
```

---

## EXAMPLE 4
**SOURCE:** tests\test_tutorial\test_server_sent_events\test_tutorial005.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_stream_chat
Arguments: []
Source_File: tests\test_tutorial\test_server_sent_events\test_tutorial005.py
Docstring/Description:
```

---

## EXAMPLE 5
**SOURCE:** tests\test_ws_router.py

**CONTENT BLOCK:**
```
Type: function
 Name: index
Arguments: []
Source_File: tests\test_ws_router.py
Docstring/Description:
```

---

## EXAMPLE 6
**SOURCE:** wikis/docs-en-docs-advanced-dataclasses.md

**CONTENT BLOCK:**
```
Context: [Using Dataclasses { #using-dataclasses } > Dataclasses in Nested Data Structures { #dataclasses-in-nested-data-structures }]
 Content: You can also combine `dataclasses` with other type annotations to make nested data structures.  
In some cases, you might still have to use Pydantic's version of `dataclasses`. For example, if you have errors with the automatically generated API documentation.  
In that case, you can simply swap the standard `dataclasses` with `pydantic.dataclasses`, which is a drop-in replacement:  
{* ../../docs_src/dataclasses_/tutorial003_py310.py hl[1,4,7:10,13:16,22:24,27] *}  
1. We still import `field` from standard `dataclasses`.  
2. `pydantic.dataclasses` is a drop-in replacement for `dataclasses`.  
3. The `Author` dataclass includes a list of `Item` dataclasses.  
4. The `Author` dataclass is used as the `response_model` parameter.  
5. You can use other standard type annotations with dataclasses as the request body.  
In this case, it's a list of `Item` dataclasses.  
6. Here we are returning a dictionary that contains `items` which is a list of dataclasses.  
FastAPI is still capable of <dfn title="converting the data to a format that can be transmitted">serializing</dfn> the data to JSON.  
7. Here the `response_model` is using a type annotation of a list of `Author` dataclasses.  
Again, you can combine `dataclasses` with standard type annotations.  
8. Notice that this *path operation function* uses regular `def` instead of `async def`.  
As always, in FastAPI you can combine `def` and `async def` as needed.  
If you need a refresher about when to use which, check out the section _"In a hurry?"_ in the docs about [`async` and `await`](../async.md#in-a-hurry).  
9. This *path operation function* is not returning dataclasses (although it could), but a list of dictionaries with internal data.  
FastAPI will use the `response_model` parameter (that includes dataclasses) to convert the response.  
You can combine `dataclasses` with other type annotations in many different combinations to form complex data structures.  
Check the in-code annotation tips above to see more specific details.
```

---

## EXAMPLE 7
**SOURCE:** fastapi\routing.py

**CONTENT BLOCK:**
```
Type: function
 Name: __init__
Arguments: []
Source_File: fastapi\routing.py
Docstring/Description:
```

---

## EXAMPLE 8
**SOURCE:** tests\test_jsonable_encoder.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_encode_model_with_alias
Arguments: []
Source_File: tests\test_jsonable_encoder.py
Docstring/Description:
```

---

## EXAMPLE 9
**SOURCE:** wikis/docs-en-docs-release-notes.md

**CONTENT BLOCK:**
```
Context: [Release Notes > 0.118.1 (2025-10-08) > Upgrades]
 Content: * 👽️ Ensure compatibility with Pydantic 2.12.0. PR [#14036](https://github.com/fastapi/fastapi/pull/14036) by [@cjwatson](https://github.com/cjwatson).
```

---

## EXAMPLE 10
**SOURCE:** wikis/docs-en-docs-tutorial-cors.md

**CONTENT BLOCK:**
```
Context: [CORS (Cross-Origin Resource Sharing) { #cors-cross-origin-resource-sharing } > Use `CORSMiddleware` { #use-corsmiddleware } > Simple requests { #simple-requests }]
 Content: Any request with an `Origin` header. In this case the middleware will pass the request through as normal, but will include appropriate CORS headers on the response.
```

---

## EXAMPLE 11
**SOURCE:** tests\test_schema_extra_examples.py

**CONTENT BLOCK:**
```
Type: function
 Name: query_examples
Arguments: []
Source_File: tests\test_schema_extra_examples.py
Docstring/Description:
```

---

## EXAMPLE 12
**SOURCE:** tests\test_pydanticv2_dataclasses_uuid_stringified_annotations.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_annotations
Arguments: []
Source_File: tests\test_pydanticv2_dataclasses_uuid_stringified_annotations.py
Docstring/Description:
```

---

## EXAMPLE 13
**SOURCE:** scripts\tests\test_translation_fixer\test_code_includes\test_number_mismatch.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_gt
Arguments: []
Source_File: scripts\tests\test_translation_fixer\test_code_includes\test_number_mismatch.py
Docstring/Description:
```

---

## EXAMPLE 14
**SOURCE:** tests\test_response_model_as_return_annotation.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_response_model_no_annotation_return_submodel_with_extra_data
Arguments: []
Source_File: tests\test_response_model_as_return_annotation.py
Docstring/Description:
```

---

## EXAMPLE 15
**SOURCE:** tests\test_validate_response.py

**CONTENT BLOCK:**
```
Type: function
 Name: get_innerinvalid
Arguments: []
Source_File: tests\test_validate_response.py
Docstring/Description:
```

---

## EXAMPLE 16
**SOURCE:** tests\test_tutorial\test_schema_extra_example\test_tutorial003.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_openapi_schema
Arguments: []
Source_File: tests\test_tutorial\test_schema_extra_example\test_tutorial003.py
Docstring/Description:
```

---

## EXAMPLE 17
**SOURCE:** wikis/docs-en-docs-tutorial-body.md

**CONTENT BLOCK:**
```
Context: [Request Body { #request-body } > Editor support { #editor-support }]
 Content: In your editor, inside your function you will get type hints and completion everywhere (this wouldn't happen if you received a `dict` instead of a Pydantic model):  
<img src="/img/tutorial/body/image03.png">  
You also get error checks for incorrect type operations:  
<img src="/img/tutorial/body/image04.png">  
This is not by chance, the whole framework was built around that design.  
And it was thoroughly tested at the design phase, before any implementation, to ensure it would work with all the editors.  
There were even some changes to Pydantic itself to support this.  
The previous screenshots were taken with [Visual Studio Code](https://code.visualstudio.com).  
But you would get the same editor support with [PyCharm](https://www.jetbrains.com/pycharm/) and most of the other Python editors:  
<img src="/img/tutorial/body/image05.png">  
/// tip  
If you use [PyCharm](https://www.jetbrains.com/pycharm/) as your editor, you can use the [Pydantic PyCharm Plugin](https://github.com/koxudaxi/pydantic-pycharm-plugin/).  
It improves editor support for Pydantic models, with:  
* auto-completion
* type checks
* refactoring
* searching
* inspections  
///
```

---

## EXAMPLE 18
**SOURCE:** tests\test_request_params\test_body\test_optional_str.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_optional_alias_and_validation_alias_schema
Arguments: []
Source_File: tests\test_request_params\test_body\test_optional_str.py
Docstring/Description:
```

---

## EXAMPLE 19
**SOURCE:** wikis/docs-en-docs-release-notes.md

**CONTENT BLOCK:**
```
Context: [Release Notes > 0.57.0 (2020-06-13)]
 Content: * Remove broken link from "External Links". PR [#1565](https://github.com/tiangolo/fastapi/pull/1565) by [@victorphoenix3](https://github.com/victorphoenix3).
* Update/fix docs for [WebSockets with dependencies](https://fastapi.tiangolo.com/advanced/websockets/#using-depends-and-others). Original PR [#1540](https://github.com/tiangolo/fastapi/pull/1540) by [@ChihSeanHsu](https://github.com/ChihSeanHsu).
* Add support for Python's `http.HTTPStatus` in `status_code` parameters. PR [#1534](https://github.com/tiangolo/fastapi/pull/1534) by [@retnikt](https://github.com/retnikt).
* When using Pydantic models with `__root__`, use the internal value in `jsonable_encoder`. PR [#1524](https://github.com/tiangolo/fastapi/pull/1524) by [@patrickkwang](https://github.com/patrickkwang).
* Update docs for path parameters. PR [#1521](https://github.com/tiangolo/fastapi/pull/1521) by [@yankeexe](https://github.com/yankeexe).
* Update docs for first steps, links and rewording. PR [#1518](https://github.com/tiangolo/fastapi/pull/1518) by [@yankeexe](https://github.com/yankeexe).
* Enable `showCommonExtensions` in Swagger UI to show additional validations like `maxLength`, etc. PR [#1466](https://github.com/tiangolo/fastapi/pull/1466) by [@TiewKH](https://github.com/TiewKH).
* Make `OAuth2PasswordRequestFormStrict` importable directly from `fastapi.security`. PR [#1462](https://github.com/tiangolo/fastapi/pull/1462) by [@RichardHoekstra](https://github.com/RichardHoekstra).
* Add docs about [Default response class](https://fastapi.tiangolo.com/advanced/custom-response/#default-response-class). PR [#1455](https://github.com/tiangolo/fastapi/pull/1455) by [@TezRomacH](https://github.com/TezRomacH).
* Add note in docs about additional parameters `response_model_exclude_defaults` and `response_model_exclude_none` in [Response Model](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter). PR [#1427](https://github.com/tiangolo/fastapi/pull/1427) by [@wshayes](https://github.com/wshayes).
* Add note about [PyCharm Pydantic plugin](https://github.com/koxudaxi/pydantic-pycharm-plugin) to docs. PR [#1420](https://github.com/tiangolo/fastapi/pull/1420) by [@koxudaxi](https://github.com/koxudaxi).
* Update and clarify testing function name. PR [#1395](https://github.com/tiangolo/fastapi/pull/1395) by [@chenl](https://github.com/chenl).
* Fix duplicated headers created by indirect dependencies that use the request directly. PR [#1386](https://github.com/tiangolo/fastapi/pull/1386) by [@obataku](https://github.com/obataku) from tests by [@scottsmith2gmail](https://github.com/scottsmith2gmail).
* Upgrade Starlette version to `0.13.4`. PR [#1361](https://github.com/tiangolo/fastapi/pull/1361) by [@rushton](https://github.com/rushton).
* Improve error handling and feedback for requests with invalid JSON. PR [#1354](https://github.com/tiangolo/fastapi/pull/1354) by [@aviramha](https://github.com/aviramha).
* Add support for declaring metadata for tags in OpenAPI. New docs at [Tutorial - Metadata and Docs URLs - Metadata for tags](https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-tags). PR [#1348](https://github.com/tiangolo/fastapi/pull/1348) by [@thomas-maschler](https://github.com/thomas-maschler).
* Add basic setup for Russian translations. PR [#1566](https://github.com/tiangolo/fastapi/pull/1566).
* Remove obsolete Chinese articles after adding official community translations. PR [#1510](https://github.com/tiangolo/fastapi/pull/1510) by [@waynerv](https://github.com/waynerv).
* Add `__repr__` for *path operation function* parameter helpers (like `Query`, `Depends`, etc) to simplify debugging. PR [#1560](https://github.com/tiangolo/fastapi/pull/1560) by [@rkbeatss](https://github.com/rkbeatss) and [@victorphoenix3](https://github.com/victorphoenix3).
```

---

## EXAMPLE 20
**SOURCE:** tests\test_request_params\test_form\test_optional_str.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_optional_validation_alias_schema
Arguments: []
Source_File: tests\test_request_params\test_form\test_optional_str.py
Docstring/Description:
```

---

## EXAMPLE 21
**SOURCE:** tests\test_response_model_data_filter.py

**CONTENT BLOCK:**
```
Type: function
 Name: create_user
Arguments: []
Source_File: tests\test_response_model_data_filter.py
Docstring/Description:
```

---

## EXAMPLE 22
**SOURCE:** tests\test_compat.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_model_field_default_required
Arguments: []
Source_File: tests\test_compat.py
Docstring/Description:
```

---

## EXAMPLE 23
**SOURCE:** tests\test_response_model_include_exclude.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_nested_include_mixed_dict
Arguments: []
Source_File: tests\test_response_model_include_exclude.py
Docstring/Description:
```

---

## EXAMPLE 24
**SOURCE:** tests\test_request_params\test_cookie\test_required_str.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_required_alias_by_alias
Arguments: []
Source_File: tests\test_request_params\test_cookie\test_required_str.py
Docstring/Description:
```

---

## EXAMPLE 25
**SOURCE:** wikis/docs-en-docs-advanced-generate-clients.md

**CONTENT BLOCK:**
```
Context: [Generating SDKs { #generating-sdks } > Custom Operation IDs and Better Method Names { #custom-operation-ids-and-better-method-names }]
 Content: You can **modify** the way these operation IDs are **generated** to make them simpler and have **simpler method names** in the clients.  
In this case, you will have to ensure that each operation ID is **unique** in some other way.  
For example, you could make sure that each *path operation* has a tag, and then generate the operation ID based on the **tag** and the *path operation* **name** (the function name).
```

---

## EXAMPLE 26
**SOURCE:** tests\test_invalid_path_param.py

**CONTENT BLOCK:**
```
Type: function
 Name: test_invalid_simple_dict
Arguments: []
Source_File: tests\test_invalid_path_param.py
Docstring/Description:
```

---

## EXAMPLE 27
**SOURCE:** wikis/docs-en-docs-advanced-settings.md

**CONTENT BLOCK:**
```
Context: [Settings and Environment Variables { #settings-and-environment-variables } > Reading a `.env` file { #reading-a-env-file } > Creating the `Settings` only once with `lru_cache` { #creating-the-settings-only-once-with-lru-cache }]
 Content: Reading a file from disk is normally a costly (slow) operation, so you probably want to do it only once and then reuse the same settings object, instead of reading it for each request.  
But every time we do:  
```Python
Settings()
```  
a new `Settings` object would be created, and at creation it would read the `.env` file again.  
If the dependency function was just like:  
```Python
def get_settings():
return Settings()
```  
we would create that object for each request, and we would be reading the `.env` file for each request. ⚠️  
But as we are using the `@lru_cache` decorator on top, the `Settings` object will be created only once, the first time it's called. ✔️  
{* ../../docs_src/settings/app03_an_py310/main.py hl[1,11] *}  
Then for any subsequent call of `get_settings()` in the dependencies for the next requests, instead of executing the internal code of `get_settings()` and creating a new `Settings` object, it will return the same object that was returned on the first call, again and again.
```

---

## EXAMPLE 28
**SOURCE:** docs_src\query_param_models\tutorial002_py310.py

**CONTENT BLOCK:**
```
Type: function
 Name: read_items
Arguments: []
Source_File: docs_src\query_param_models\tutorial002_py310.py
Docstring/Description:
```

---

## EXAMPLE 29
**SOURCE:** wikis/docs-en-docs-advanced-middleware.md

**CONTENT BLOCK:**
```
Context: [Advanced Middleware { #advanced-middleware } > Integrated middlewares { #integrated-middlewares }]
 Content: **FastAPI** includes several middlewares for common use cases, we'll see next how to use them.  
/// note | Technical Details  
For the next examples, you could also use `from starlette.middleware.something import SomethingMiddleware`.  
**FastAPI** provides several middlewares in `fastapi.middleware` just as a convenience for you, the developer. But most of the available middlewares come directly from Starlette.  
///
```

---

## EXAMPLE 30
**SOURCE:** docs_src\graphql_\tutorial001_py310.py

**CONTENT BLOCK:**
```
Type: class
 Name: User
Arguments: []
Source_File: docs_src\graphql_\tutorial001_py310.py
Docstring/Description:
```

---

