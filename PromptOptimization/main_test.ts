import { assertEquals } from "@std/assert";
import { add } from "./main_old.ts";

Deno.test(function addTest() {
  assertEquals(add(2, 3), 5);
});
