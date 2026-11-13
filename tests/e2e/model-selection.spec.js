const { test, expect } = require("@playwright/test");

const numericInputs = {
  age: "58",
  resting_bp: "140",
  cholesterol: "240",
  max_hr: "120",
  oldpeak: "1.5",
};

const categoricalInputs = {
  sex: "1",
  chest_pain_type: "3",
  fasting_bs: "0",
  resting_ecg: "1",
  exercise_angina: "1",
  st_slope: "1",
};

async function signInAsClinician(page) {
  const username = process.env.PLAYWRIGHT_USERNAME;
  const password = process.env.PLAYWRIGHT_PASSWORD;
  test.skip(!username || !password, "Set PLAYWRIGHT_USERNAME and PLAYWRIGHT_PASSWORD.");

  await page.goto("/");
  await page.locator("#username").fill(username);
  await page.locator("#password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.locator("#roleChip")).toHaveText("Clinician");
  await expect(page.locator("#predictPage")).toBeVisible();
}

test("shows the secure sign-in screen before authentication", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Sign in to your workspace" })).toBeVisible();
  await expect(page.locator("#loginForm")).toBeVisible();
  await expect(page.locator("#appView")).toBeHidden();
});

test("requires both credentials before submitting sign-in", async ({ page }) => {
  await page.goto("/");

  const username = page.locator("#username");
  const password = page.locator("#password");
  await page.getByRole("button", { name: "Sign in" }).click();

  expect(await username.evaluate((input) => input.validity.valueMissing)).toBe(true);
  expect(await password.evaluate((input) => input.validity.valueMissing)).toBe(true);
  await expect(page.locator("#appView")).toBeHidden();
});

test("Clinician sees the available active model versions", async ({ page }) => {
  await signInAsClinician(page);

  const modelSelect = page.getByLabel("Prediction model");
  await expect(page.locator("#modelAvailabilityNote")).toContainText(
    "Only active, validated versions are selectable."
  );
  const versions = await modelSelect.locator("option").evaluateAll((options) =>
    options.map((option) => option.value).filter(Boolean)
  );
  expect(versions.length).toBeGreaterThan(0);
  expect(new Set(versions).size).toBe(versions.length);
});

test("requires all clinical inputs and enforces measurement bounds", async ({ page }) => {
  await signInAsClinician(page);

  const clinicalInputs = page.locator(
    "#predictionForm input[required], #predictionForm select[required]:not(#model)"
  );
  await expect(clinicalInputs).toHaveCount(11);
  expect(await clinicalInputs.evaluateAll((inputs) => inputs.every((input) => input.required))).toBe(true);

  const age = page.locator("#f_age");
  await age.fill("0");
  expect(await age.evaluate((input) => input.validity.rangeUnderflow)).toBe(true);
  await age.fill("121");
  expect(await age.evaluate((input) => input.validity.rangeOverflow)).toBe(true);
  await age.fill("58.5");
  expect(await age.evaluate((input) => input.validity.stepMismatch)).toBe(true);
});

test("theme toggle switches between dark and light", async ({ page }) => {
  await signInAsClinician(page);

  const root = page.locator("html");
  const initialTheme = (await root.getAttribute("data-theme")) || "light";
  await page.getByRole("button", { name: "Toggle color theme" }).click();
  await expect(root).toHaveAttribute("data-theme", initialTheme === "dark" ? "light" : "dark");
  await page.getByRole("button", { name: "Toggle color theme" }).click();
  await expect(root).toHaveAttribute("data-theme", initialTheme);
});

test("sign out returns the user to the login screen", async ({ page }) => {
  await signInAsClinician(page);

  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.locator("#loginView")).toBeVisible();
  await expect(page.locator("#appView")).toBeHidden();
  await expect(page.locator("#password")).toHaveValue("");
});

test("Clinician can predict with each active model version", async ({ page }) => {
  await signInAsClinician(page);

  const modelSelect = page.getByLabel("Prediction model");
  await expect(page.locator("#modelAvailabilityNote")).toContainText(
    "Only active, validated versions are selectable."
  );
  const versions = await modelSelect.locator("option").evaluateAll((options) =>
    options.map((option) => option.value).filter(Boolean)
  );
  expect(versions.length).toBeGreaterThan(0);
  for (const version of versions) {
    await modelSelect.selectOption(version);
    for (const [field, value] of Object.entries(numericInputs)) {
      await page.locator(`#f_${field}`).fill(value);
    }
    for (const [field, value] of Object.entries(categoricalInputs)) {
      await page.locator(`#f_${field}`).selectOption(value);
    }
    await page.getByRole("button", { name: "Run prediction" }).click();
    await expect(page.locator("#resultModel")).toHaveText(`Model version: ${version}`);
  }
});
