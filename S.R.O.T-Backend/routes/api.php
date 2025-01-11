<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\BuildingController;
use App\Http\Controllers\FloorController;
use App\Http\Controllers\GenderController;
use App\Http\Controllers\UserController;
use App\Http\Controllers\ToiletController;
use App\Http\Controllers\LoggingCounterController;
use App\Http\Controllers\LoginController;
use App\Http\Controllers\UsageLogController;

Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

// Building Routes
Route::get('/buildings', [BuildingController::class, 'index']); // List all buildings
Route::post('/buildings', [BuildingController::class,'store']); // Create a new building

// Floor Routes
Route::get('/buildings/{building_id}/floors', [FloorController::class, 'index']); // List floors within a specific building
Route::post('/buildings/{building_id}/floors', [FloorController::class, 'addFloor']); // Add a floor with toilets to a specific building
Route::delete('/floors/{id}', [FloorController::class, 'destroy']); // Delete a specific floor

// Gender Routes
Route::get('/genders', [GenderController::class, 'index']); // List all genders
Route::get('/genders/{id}', [GenderController::class, 'show']); // Show a specific gender
Route::post('/genders', [GenderController::class, 'store']); // Create a new gender
Route::put('/genders/{id}', [GenderController::class, 'update']); // Update a specific gender
Route::delete('/genders/{id}', [GenderController::class, 'destroy']); // Delete a specific gender

// User Routes
Route::get('/users', [UserController::class, 'index']); // List all users
Route::get('/users/{id}', [UserController::class, 'show']); // Show a specific user
Route::post('/users', [UserController::class, 'store']); // Create a new user
Route::put('/users/{id}', [UserController::class, 'update']); // Update a specific user
Route::delete('/users/{id}', [UserController::class, 'destroy']); // Delete a specific user

// Usage Log Routes
Route::get('/usagelogs', [UsageLogController::class, 'index']); // List all usage logs
Route::get('/usagelogs/{id}', [UsageLogController::class, 'show']); // Show a specific usage log
Route::post('/usagelogs', [UsageLogController::class, 'store']); // Create a new usage log entry
Route::delete('/usagelogs/{id}', [UsageLogController::class, 'destroy']); // Delete a specific usage log

// Toilet Routes
Route::prefix('floors/{floor}/toilets')->group(function () {
    Route::get('/', [ToiletController::class, 'index']); // Get all toilets for a specific floor
    Route::post('/', [ToiletController::class, 'store']); // Create a new toilet
    Route::get('/{toilet}', [ToiletController::class, 'show']); // Get a specific toilet
    Route::put('/{toilet}', [ToiletController::class, 'update']); // Update a specific toilet
    Route::delete('/{toilet}', [ToiletController::class, 'destroy']); // Delete a specific toilet
    Route::put('/{toilet}/reset', [ToiletController::class, 'resetUsageCount']); // Reset usage count for a specific toilet
});

// Data Logging Route
Route::get('/logging-counters', [LoggingCounterController::class, 'index']); // List all logs
Route::post('/logging-counters/manual', [LoggingCounterController::class, 'logManual']);
Route::post('/toilets/{toiletId}/log', [LoggingCounterController::class, 'logSpecificToilet']);
Route::put('/toilets/{toiletId}/reset', [LoggingCounterController::class, 'resetUsageCount']); // Reset and log action
Route::delete('/logging-counters/clear', [LoggingCounterController::class, 'clearAllLogs']); // Clear all logs
Route::delete('/logging-counters/{id}', [LoggingCounterController::class, 'deleteSpecificLog']); // Delete a specific log

// Login Route
Route::post('/login', [LoginController::class, 'login']); // Login route
Route::post('/logout', [LoginController::class, 'logout']); // Logout route
